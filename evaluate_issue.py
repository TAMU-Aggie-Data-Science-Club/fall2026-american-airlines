"""Score a previously frozen forecast against a later, locally prepared panel."""
import argparse,io,json,hashlib,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def evaluate(issue,panel):
    with zipfile.ZipFile(issue) as z:
        manifest=json.loads(z.read('manifest.json'));payload=z.read('market_forecast.csv')
    if hashlib.sha256(payload).hexdigest()!=manifest['forecast_sha256']:raise ValueError('Frozen forecast hash mismatch')
    forecast=pd.read_csv(io.BytesIO(payload),parse_dates=['date'])
    data=np.load(panel,allow_pickle=False);dates=pd.DatetimeIndex(data['dates']);markets=data['markets'];values=data['values']
    target_dates=pd.DatetimeIndex(sorted(forecast.date.unique()));available=target_dates.intersection(dates)
    result={'issue_id':manifest['issue_id'],'forecast_origin':manifest['forecast_origin'],'available_target_months':len(available),'required_target_months':len(target_dates),'status':'pending actuals' if not len(available) else 'partial observed evaluation' if len(available)<len(target_dates) else 'complete calendar window; verify reporting completeness'}
    if not len(available):return result
    abs_error=0.;actual_sum=0.;known=0;missing=0;unforecast_actual=0.
    for date in available:
        truth=pd.Series(values[dates.get_loc(date)],index=markets)
        f=forecast[forecast.date==date].set_index('market').forecast_passengers
        valid=truth.dropna();prediction=f.reindex(valid.index).fillna(0)
        abs_error+=float(abs(prediction-valid).sum());actual_sum+=float(valid.sum());known+=len(valid)
        missing+=int(truth.reindex(f.index).isna().sum())
        unforecast_actual+=float(valid[~valid.index.isin(f.index)].sum())
    result.update({'market_wape_pct':100*abs_error/actual_sum if actual_sum else None,'observed_market_months':known,'forecast_market_months_without_actuals':missing,'unforecast_actual_demand':unforecast_actual,
                   'warning':'Calendar presence does not prove complete reporting. No missing actual is converted to zero. Partial results do not establish full second-year accuracy.'})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--issue',default='data/processed/results/forecast_issue.zip');p.add_argument('--panel',default='data/processed/panel.npz');p.add_argument('--output',default='data/processed/results/prospective_evaluation.json');a=p.parse_args()
    r=evaluate(a.issue,a.panel);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
