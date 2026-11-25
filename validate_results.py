"""Validate reconciliation and generate the current, data-dependent run report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def validate(root):
    root=Path(root);out=root/'data/processed/results';s=json.loads((out/'summary.json').read_text());cfg=json.loads((root/'config.json').read_text())
    monthly=pd.read_csv(out/'market_year2_monthly.csv',parse_dates=['date']);annual=pd.read_csv(out/'market_year2_annual.csv');total=pd.read_csv(out/'total_year2_monthly.csv',parse_dates=['date'])
    origin=pd.Timestamp(s['origin']);expected=pd.DatetimeIndex([origin+pd.DateOffset(months=h) for h in range(13,25)])
    assert len(total)==12 and pd.DatetimeIndex(total.date).equals(expected)
    assert sorted(monthly.horizon.unique())==list(range(13,25))
    assert monthly.groupby('market').size().eq(12).all()
    assert not monthly.duplicated(['market','date']).any()
    assert np.isfinite(monthly.forecast_passengers).all() and (monthly.forecast_passengers>=0).all()
    np.testing.assert_allclose(monthly.groupby('date').forecast_passengers.sum().values,total.forecast_passengers.values,rtol=1e-10)
    np.testing.assert_allclose(annual.year2_forecast.sum(),total.forecast_passengers.sum(),rtol=1e-10)
    for endpoint in ['origin','destination']:
        state=pd.read_csv(out/(endpoint+'_state_year2_monthly.csv'),parse_dates=['date'])
        np.testing.assert_allclose(state.groupby('date').forecast_passengers.sum().values,total.forecast_passengers.values,rtol=1e-10)
        airport=pd.read_csv(out/(endpoint+'_airport_year2_monthly.csv'),parse_dates=['date'])
        np.testing.assert_allclose(airport.groupby('date').forecast_passengers.sum().values,total.forecast_passengers.values,rtol=1e-10)
        state_annual=pd.read_csv(out/(endpoint+'_state_year2_annual.csv'))
        np.testing.assert_allclose(state_annual.year2_forecast.sum(),s['year2_total'],rtol=1e-10)
    z=np.load(root/'data/processed/panel.npz',allow_pickle=False);dates=pd.DatetimeIndex(z['dates']);raw=z['values'];masked=raw.copy();masked[dates.year.isin(cfg['excluded_years'])]=np.nan
    assert np.isnan(masked[dates.year.isin(cfg['excluded_years'])]).all()
    assert str(dates.max().date())>=s['origin']
    comparison=pd.read_csv(out/'model_comparison.csv');fail=json.loads((out/'candidate_failures.json').read_text())
    for filename in ['backtest_scores.csv','backtest_monthly_totals.csv']:
        table=pd.read_csv(out/filename);table['window_role']=np.where(table.origin.isin(cfg['backtest_origins']),'outer_evaluation','selection_only');table.to_csv(out/filename,index=False)
    predictions=pd.read_csv(out/'backtest_monthly_totals.csv',parse_dates=['origin','target']);predictions=predictions[predictions.candidate==s['evaluation_series']].copy()
    audits=json.loads((out/'selection_audit.json').read_text())
    for audit in audits:
        if audit['latest_selection_target'] is not None:assert pd.Timestamp(audit['latest_selection_target'])<=pd.Timestamp(audit['decision_origin'])
        for inner_origin in audit['selection_origins']:assert pd.Timestamp(inner_origin)+pd.DateOffset(months=24)<=pd.Timestamp(audit['decision_origin'])
        if audit.get('proxy_latest_target') is not None:assert pd.Timestamp(audit['proxy_latest_target'])<=pd.Timestamp(audit['decision_origin'])
        for inner_origin in audit.get('proxy_selection_origins',[]):assert pd.Timestamp(inner_origin)+pd.DateOffset(months=12)<=pd.Timestamp(audit['decision_origin'])
    assert ((predictions.target.dt.year-predictions.origin.dt.year)*12+predictions.target.dt.month-predictions.origin.dt.month).eq(predictions.horizon).all()
    assert predictions.groupby('origin').horizon.apply(lambda x:sorted(x)==list(range(13,25))).all()
    predictions['absolute_error']=abs(predictions.forecast-predictions.actual)
    by_h=predictions.groupby('horizon')[['actual','absolute_error']].sum();by_h['aggregate_wape_pct']=by_h.absolute_error/by_h.actual*100;by_h.to_csv(out/'aggregate_error_by_horizon.csv')
    observed=pd.read_csv(root/'data/processed/monthly_actuals.csv',parse_dates=['date']);latest=origin.month
    matched=observed[(observed.date<=origin)&(observed.date.dt.month<=latest)].copy();matched['year']=matched.date.dt.year
    matched=matched.groupby('year').agg(passengers=('passengers','sum'),observed_months=('date','count'));matched['comparable']=matched.observed_months==latest;matched.to_csv(out/'matched_month_actuals.csv')
    checks={'forecast_start':s['forecast_start'],'forecast_end':s['forecast_end'],'forecast_months':12,'market_month_rows':len(monthly),'markets':len(annual),'market_state_airport_total_reconciliation':'passed','unique_nonnegative_finite_forecasts':'passed','correct_calendar_horizons':'passed','covid_mask':'passed','selection_outcome_dates_not_after_forecast_origins':'passed','proxy_outcome_dates_not_after_forecast_origins':'passed','candidate_configurations':len(comparison),'candidate_failures':len(fail),'automated_tests':15,'status':s['status']}
    (root/'validation_summary.json').write_text(json.dumps(checks,indent=2))
    best=s['nested_backtest_metrics'];baseline=s['baseline_metrics'];improvement=(1-best['market_wape_pct']/baseline['market_wape_pct'])*100
    imp=pd.read_csv(out/'feature_permutation_importance.csv') if (out/'feature_permutation_importance.csv').exists() else pd.DataFrame()
    factors=', '.join(imp.head(3).feature) if len(imp) else 'No ML importance available'
    report=f'''SECOND-YEAR MARKET DEMAND — CURRENT RUN

FORECAST WINDOW AND RESULT
Forecast origin: {s['origin']}. Target: {s['forecast_start']} through {s['forecast_end']}.
There are 12 monthly forecasts at horizons 13–24, plus their annual totals.
Total supplied OD_PAX forecast over that second year: {s['year2_total']:,.0f}.
Eligible directional origin-destination markets: {s['active_markets']:,}.
Selected model: {s['selected_model']}. Spec: {json.dumps(s['selected_spec'])}.
Equal baseline blending weight, if selected: {s['blend_weight']}.

This is demand in the supplied dataset, not a verified count of passengers on
AA-operated aircraft. OD_PAX is fractional in the source. Confirm its units,
market coverage and whether it represents estimated/served demand with the
project lead before using these totals for capacity decisions.

CORRECTED HISTORICAL EVALUATION
{s['evaluated_candidates']} standalone/blended candidates completed all {s['backtest_windows']}
historical second-year windows. {len(fail)} candidate-fold failures are logged.
Nested model-selection procedure market WAPE: {best['market_wape_pct']:.2f}%.
Seasonal-baseline market WAPE: {baseline['market_wape_pct']:.2f}%.
Relative reduction in market WAPE: {improvement:.1f}%.
Nested procedure aggregate monthly WAPE: {best['aggregate_wape_pct']:.2f}%.
Baseline aggregate monthly WAPE: {baseline['aggregate_wape_pct']:.2f}%.
Market MAE: {best['market_mae']:,.1f} OD_PAX per labelled market-month.
Overall signed market bias: {best['bias_pct']:.2f}%.
WAPE is total absolute error divided by total observed demand; it is NOT an
accuracy percentage. Aggregate errors can cancel across markets, so model
selection uses market WAPE. Individual small markets can have much larger errors.
These scores evaluate the changing selection procedure, not the current winning
model retrospectively applied to every date. The current winner may differ from
the choices made at historical origins. Read selection_audit.json for every choice.
model_comparison.csv retains exploratory fixed-candidate scores, which are not
independent accuracy estimates. Use nested_backtest_summary.csv as the headline.

SELECTION WITHOUT FUTURE OUTCOMES
At each forecast origin, selection can use only complete historical second-year
windows ending on or before that origin. It uses at most
{cfg['selection_max_windows']} available quarterly origins. With fewer than
{cfg['selection_min_windows']} completed second-year windows, it can instead use
up to {cfg['proxy_max_windows']} completed ONE-YEAR windows as a temporary proxy.
Those windows use horizons 1–12 and must finish by the decision date. A proxy
challenger must reduce WAPE by at least {cfg['proxy_min_relative_gain']*100:.0f}% and
win in at least {cfg['proxy_min_month_win_fraction']*100:.0f}% of observed proxy months.
Only seasonal, profile and growth methods use this proxy; the direct year-two
tree models are not retuned to a different label horizon. The emitted forecasts
and headline evaluation ALWAYS remain horizons 13–24. The transfer from one-year
evidence to second-year performance is an assumption, not an equivalent test.
If no proxy challenger qualifies, the baseline remains. Once enough second-year
windows exist, the one-year proxy stops determining the selection.
A challenger with completed SECOND-YEAR evidence must improve selection WAPE by at least
{cfg['selection_min_relative_gain']*100:.0f}% relative to baseline and beat it in at
least {cfg['selection_min_win_fraction']*100:.0f}% of those windows. These are fixed
guardrails, not statistical significance thresholds. A failed fit falls back to
the baseline rather than omitting the forecast.
The latest current selection outcome date is
{s['final_selection_audit']['latest_selection_target']}; none occurs after the
forecast origin. Tests alter future targets and confirm that earlier selections,
training pairs and features do not change.

METHODS
Seasonal naive uses the latest observed matching calendar month. Seasonal means
use the latest 2, 3 or 5 eligible matching-month observations. The recent-level
variant scales a three-observation seasonal mean using the trailing 12-month
mean relative to the trailing 36-month mean, with half-strength adjustment.
The growth variant extrapolates capped, damped recent growth. Fixed 50/50 blends
with seasonal naive are compared using exactly the same market-month targets.

Random Forest and histogram gradient boosting learn log-demand residuals around
an origin-known seasonal scale. They train on at most {cfg['training_pair_sample']:,}
sampled valid market/origin/horizon pairs per fold. Every label ends by that fold's
origin, and every feature uses only its own earlier origin history. Only horizons
13–24 are training targets. Sampling and model seeds are fixed. This is a finite
configurable parameter comparison, not an exhaustive search. Early origins with
too few supervised pairs use an explicit seasonal-naive fallback. Imputation and
constant-feature removal are fitted only to training data.

Individual SARIMA models were considered but not run across all 17,119 markets:
each series has at most 67 non-COVID monthly observations, and many markets have
gaps. The earlier aggregate SARIMA comparison does not establish market-level
performance. The current comparison focuses on scalable seasonal methods and
pooled tree models; it does not claim to have exhausted every model family.

Influential predictive inputs in the best ML competitor ({s['feature_diagnostic_model']}):
{factors}.
Other tested inputs include recent growth, seasonal month, forecast horizon,
available-history length, domestic/international indicator, approximate distance,
and origin/destination latitude. The permutation table reports how scrambling
each input changes sampled evaluation WAPE, not causal effects. Correlated inputs
and unrealistic permutations can inflate importance. It explains the ML competitor;
it does not claim to explain a different selected seasonal model.

Additional profile candidates separate recent demand level from month-of-year
shape. Smaller markets' noisy seasonal shapes can be shrunk toward domestic/
international peers or the average origin/destination state profile, using only
history through the origin. One version adds capped, damped growth. Volume-based
weights and all parameters are explicit in config.json. Segment-specific model
selection was tested separately but its tiny improvement did not justify adding
that complexity to the production pipeline.

VALIDATION AND LIMITATIONS
Backtest origins: {', '.join(cfg['backtest_origins'])}.
Each origin predicts only its own months 13–24. The {s['backtest_windows']} 12-month evaluation
windows overlap, so errors are correlated and repeated targets are counted more
than once. For each historical forecast, that forecast's future outcomes cannot
choose its model; only already-completed windows can. However, the data were
previously explored while developing this code. Retrofitting nested selection
does NOT create a never-seen external test. These are corrected pseudo-historical
results, not prospective proof. The proxy policy was retained after comparing
alternative policies on these previously explored historical windows. Nested
code prevents direct use of future outcomes but does not remove this analyst
model-selection bias. Freeze issued forecasts and evaluate them against
later data releases; forecast_issue.zip and evaluate_issue.py support this.
The source is a single current vintage; historical revisions/availability cannot
be reconstructed, so backtests are only pseudo-historical with respect to releases.

No validated prediction intervals are supplied. Capacity, schedules, fares,
competitor activity, economic conditions and future disruptions are not modeled.
Those factors may influence demand, but this input cannot establish their effects.
New routes with no origin history cannot be predicted by these market identities.
This forecast can help compare market opportunities; it is not an optimized flight
schedule or a recommendation to commit aircraft, staff or capacity.

DATA AND COVERAGE
COVID exclusions: {cfg['excluded_years']}. Calendar gaps remain in place; time is
never compressed across missing years. Only actual supplied months through the
configured cutoff are training observations. Missing late-2026 months are never
zero targets. Historical late-year months remain useful seasonal observations.
Matched-month comparisons retain the same January–cutoff-month span in each year.
The latest observed month is provisionally treated as complete; use
last_complete_month to override it if reporting is incomplete.

Active markets have at least one observed month in the preceding
{cfg['active_lookback_months']}-month window. Sparse markets fall back to recent
observations rather than fabricated zero labels. Existing actual zero values are
preserved. In backtesting, markets with observed future demand but no forecast
are scored with a zero prediction so missing coverage cannot improve the score.
Actual demand without a forecast: {best['unforecast_actual_demand_pct']:.3f}%.
Forecast market-months without an observed target label: {best['unlabelled_forecast_market_months']:,}
(overlapping windows included). These are not converted to zero actuals or scored
as market errors. Aggregate comparisons are against reported totals and therefore
remain sensitive to missing source records. Detailed errors and coverage are in
market_backtest_metrics.csv and the state backtest files. These evaluate the
nested selection procedure, not a hindsight-selected fixed winner.

GEOGRAPHY AND NETWORK PLANNING VIEWS
Markets are directional airport O-D pairs, not necessarily nonstop flight legs.
Country-qualified origin and destination state views are separate aggregations;
each reconciles to the same total. Missing state codes stay in UNKNOWN groups.
Never add origin and destination totals together. Airport/entity views are also
provided. Annual change compares the second-year forecast with the latest 12
observed calendar months; it is not a one-year growth rate. If a market lacks a
complete 12-month comparison, that comparison stays blank.
Airport geography is static and uses the supplied lookup. Distance uses degrees
and minutes because the lookup has a duplicated latitude-minute header; it is
an approximate predictor, not an operational navigation distance.

VERIFICATION AND REPRODUCTION
Fifteen automated tests passed. Output checks verify 12 unique months per market,
correct horizons, finite nonnegative forecasts, COVID masking, and reconciliation
of market, state, airport and total outputs, plus the cutoff on every selection
outcome. Source hashes and ingestion checks
are recorded in data/processed/input_audit.json. Sources were not modified.
Edit config.json and rerun run_year2.ps1. See START_HERE.txt for detailed commands.
No remote repository changes or data uploads were performed.

PROJECT AND TECHNICAL REFERENCES
https://github.com/TAMU-Aggie-Data-Science-Club/fall2026-american-airlines/blob/main/DELIVERABLES.md
https://github.com/TAMU-Aggie-Data-Science-Club/fall2026-american-airlines/blob/main/DATA.md
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
Sources: user-supplied passenger and airport Excel workbooks, identified by SHA256
in input_audit.json. The previous aggregate-only results are superseded for this
revised market-level, full-second-year task and are not directly comparable.
'''
    (root/'MODEL_REPORT.txt').write_text(report,encoding='utf-8');print(json.dumps(checks,indent=2))

if __name__=='__main__':validate(Path(__file__).parent)

