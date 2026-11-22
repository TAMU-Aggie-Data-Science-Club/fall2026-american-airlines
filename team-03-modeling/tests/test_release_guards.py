"""Synthetic regressions for the integration review's scientific/control defects."""
import json
import numpy as np
import pandas as pd
import pytest
from aaforecast import config, pipeline, metrics
from aaforecast.backtest import select_shrink, interval_table, apply_aggregate_interval
from aaforecast.config import RunConfig
from aaforecast.runs import run_directory, validate_backtest
from test_features_model import _builder
from test_features_model import SMALL_GBM


def selection_frame():
    return pd.DataFrame({"origin": pd.to_datetime(["2020-01-01"] * 2),
        "target_month": pd.to_datetime(["2021-01-01"] * 2), "y": [50., 150.],
        "anchor": [100., 100.], "raw_mix": [-1., 1.], "origin_idx": [0, 0], "horizon": [12, 12]})


def test_selection_waits_for_mature_labels():
    frame = selection_frame()
    cfg = RunConfig(shrink=0., shrink_grid=(0., 1.))
    assert select_shrink([frame], "2020-04-01", cfg) == 0.
    changed = frame.assign(y=[10000., 0.])
    assert select_shrink([changed], "2020-04-01", cfg) == 0.
    assert select_shrink([frame], "2021-02-01", cfg) == 1.
    assert select_shrink([changed], "2021-02-01", cfg) == 0.


def test_missing_targets_are_not_reported_zeros(pax_long, airports):
    sparse = pax_long.drop(pax_long[(pax_long.org == "DFW") & (pax_long.des == "DAL") & (pax_long.month == "2023-01-01")].index)
    rows = _builder(sparse, airports).origin_rows(36, horizons=(12,))
    assert rows.loc[(rows.org == "DFW") & (rows.des == "DAL"), "y"].isna().all()


def test_raw_validation_stops_before_cleaning(monkeypatch, raw_pax, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline.data, "read_pax_raw", lambda: raw_pax.assign(od_pax=-1))
    def must_not_run(**kwargs):
        raise AssertionError("cleaning ran after fatal input failure")
    monkeypatch.setattr(pipeline.data, "load_pax", must_not_run)
    with pytest.raises(ValueError, match="Raw data validation failed"):
        pipeline.step_validate()


def test_metrics_and_nonfinite_rejection():
    frame = pd.DataFrame({"y": [100., 300.], "model": [110., 270.]})
    row = metrics.summarize(frame, ["model"]).iloc[0]
    assert row.mae_model == 20
    assert np.isclose(row.rmse_model, np.sqrt(500))
    assert row.mean_error_model == -10
    with pytest.raises(ValueError, match="finite"):
        metrics.summarize(frame.assign(model=np.nan), ["model"])


def test_run_identity_failure_and_compatibility(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "PAX_FILE", tmp_path / "missing.csv")
    monkeypatch.setattr(config, "AIRPORT_FILE", tmp_path / "missing-airports.csv")
    cfg = RunConfig()
    with run_directory("backtest", cfg) as first:
        pass
    validate_backtest(first, cfg)
    cfg.network_growth = .05
    with pytest.raises(ValueError, match="differs"):
        validate_backtest(first, cfg)
    with pytest.raises(RuntimeError):
        with run_directory("backtest", cfg) as failed:
            raise RuntimeError("invented failure")
    assert json.loads((failed / "manifest.json").read_text())["status"] == "failed"
    assert failed != first


def test_source_fingerprint_ignores_checkout_line_endings_but_detects_code_changes(tmp_path):
    from aaforecast.runs import source_fingerprint
    source = tmp_path / "module.py"
    source.write_bytes(b"def value():\n    return 1\n")
    before = source_fingerprint(tmp_path)
    source.write_bytes(b"def value():\r\n    return 1\r\n")
    assert source_fingerprint(tmp_path) == before
    source.write_bytes(b"def value():\r\n    return 2\r\n")
    assert source_fingerprint(tmp_path) != before


def test_aggregate_interval_uses_log1p_transform():
    table = pd.DataFrame({"level": ["network"], "q_low": [np.log(2)], "q_high": [np.log(3)]})
    out = apply_aggregate_interval(pd.DataFrame({"forecast": [0.]}), table, "network")
    assert np.isclose(out.lower.iloc[0], 1.)
    assert np.isclose(out.upper.iloc[0], 2.)


def test_end_to_end_synthetic_backtest_and_minimal_forecast(pax_long, airports):
    from aaforecast.backtest import run_backtest
    from aaforecast.forecast import run_forecast
    cfg = RunConfig(horizons=(12,), backtest_origins=("2022-06", "2022-09"),
                    min_ttm=0, gbm_params=SMALL_GBM, minimal_forecast=True)
    builder = _builder(pax_long, airports, cfg)
    result = run_backtest(builder, cfg)
    assert len(result["predictions"]) == 60
    assert np.isfinite(result["overall"].filter(regex="^(mae|rmse)_").to_numpy()).all()
    forecast = run_forecast(builder, cfg, result["interval_table"], result["aggregate_interval_table"])
    assert len(forecast["market_forecast"]) == 30
    assert np.isfinite(forecast["market_forecast"]["forecast"]).all()


def test_future_residuals_do_not_calibrate_past_intervals():
    pred = pd.DataFrame({"fold": ["past", "future"], "origin": pd.to_datetime(["2020-01-01", "2020-04-01"]),
        "target_month": pd.to_datetime(["2021-01-01", "2021-04-01"]),
        "y": [100., 100000.], "anchor_adjust": [100., 100.], "ttm": [1000., 1000.], "horizon": [12, 12]})
    _, cov = interval_table(pred, .8)
    assert (cov.status == "unavailable").all()
    assert cov.coverage.isna().all()


def test_prior_calibration_residuals_are_used_only_when_matured():
    outer = pd.DataFrame({"fold": ["outer"], "origin": pd.to_datetime(["2024-01-01"]),
        "target_month": pd.to_datetime(["2025-01-01"]), "y": [100.], "anchor_adjust": [100.],
        "ttm": [1000.], "horizon": [12]})
    history = pd.concat([outer] * 60, ignore_index=True)
    history["origin"] = pd.Timestamp("2022-01-01")
    history["target_month"] = pd.Timestamp("2023-01-01")
    _, cov = interval_table(outer, .8, calibration_predictions=history)
    assert cov.status.iloc[0] == "evaluated" and cov.calibration_rows.iloc[0] == 60
    future = history.assign(target_month=pd.Timestamp("2024-02-01"), y=10000.)
    _, unchanged = interval_table(outer, .8, calibration_predictions=pd.concat([history, future]))
    pd.testing.assert_frame_equal(cov, unchanged)


def test_bounded_experiments_record_success_and_failure(monkeypatch, tmp_path, pax_long, airports):
    from aaforecast.experiments import run_experiments, validate_spec
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "PAX_FILE", tmp_path / "missing.csv")
    monkeypatch.setattr(config, "AIRPORT_FILE", tmp_path / "missing-airports.csv")
    cfg = RunConfig(horizons=(12,), backtest_origins=("2022-06",), min_ttm=0,
                    gbm_params=SMALL_GBM, minimal_forecast=True)
    def factory(trial_cfg):
        if trial_cfg.disabled_groups:
            raise ValueError("deliberately invalid synthetic trial")
        return _builder(pax_long, airports, trial_cfg)
    directory = run_experiments({"trials": [{"name": "valid"}, {"name": "invalid", "disabled_groups": ["fare"]}]}, cfg, factory)
    registry = json.loads((directory / "registry.json").read_text())
    assert [r["status"] for r in registry] == ["review_ready", "failed"]
    assert registry[0]["gates"]["G5"] == "awaiting_independent_review"
    with pytest.raises(ValueError, match="1..4"):
        validate_spec({"trials": [{"name": str(i)} for i in range(5)]})


def test_annual_features_never_backfill_a_future_year():
    from aaforecast.features import _annual_lookup
    table = pd.DataFrame({"airport": ["DFW"], "year": [2020], "population": [100.]})
    level, growth = _annual_lookup(table, "population", np.array(["DFW"]), 2018, 2020)
    assert np.isnan(level).all() and np.isnan(growth).all()


def test_source_guard_rejects_invented_artifacts_and_accepts_code(tmp_path):
    from aaforecast.privacy import check_file
    safe = tmp_path / "safe.py"
    safe.write_text("def square(value):\n    return value * value\n")
    assert check_file(safe, tmp_path) == []
    disguised = tmp_path / "notes.txt"
    disguised.write_text("a,b,c,d\n1,2,3,4\n5,6,7,8\n9,10,11,12\n")
    assert check_file(disguised, tmp_path)
    token = tmp_path / "credential.txt"
    token.write_text("sk-" + "x" * 40)
    assert "credential-like content" in check_file(token, tmp_path)


def test_public_airport_conversion_preserves_coordinates_and_unknowns():
    from aaforecast.external.airports import convert_reference
    from aaforecast.data import dms_to_decimal
    airports = pd.DataFrame({"id": [1, 2], "iata_code": ["ABC", "ABC"], "type": ["large_airport", "closed"],
        "name": ["Invented airport", "Old duplicate"], "iso_region": ["US-TX", "US-TX"],
        "iso_country": ["US", "US"], "elevation_ft": [100., 100.],
        "latitude_deg": [32.9, 33.], "longitude_deg": [-97.04, -98.]})
    runways = pd.DataFrame({"airport_ref": [1, 1], "length_ft": [5000., 10000.]})
    result = convert_reference(airports, runways)
    assert len(result) == 1 and result.LNGST_RUNWAY_FT_QTY.iloc[0] == 10000
    assert result.WRLD_AREA_DOT_CD.isna().all()
    reconstructed = dms_to_decimal(result.LNGTD_DEGREE_QTY, result.LNGTD_MINUTE_QTY, result.LNGTD_SECOND_QTY, result.LNGTD_HEMSPHR_CD)
    assert np.isclose(reconstructed[0], -97.04)
    historical = convert_reference(airports.iloc[:1].assign(iata_code="DJT"), runways)
    assert set(historical.AIRPRT_CD) == {"DJT", "PBI"}
    assert historical.groupby("AIRPRT_CD").size().eq(1).all()


def test_team_dictionary_adapter_is_import_safe_and_preserves_totals(pax_long, airports):
    import importlib.util
    path = config.ROOT / "useful_functions" / "Data_Organizer.py"
    spec = importlib.util.spec_from_file_location("team_dictionaries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups, routes = module.build_dictionaries(pax_long, airports)
    total = sum(month["OD_PAX"] for years in routes.values() for months in years.values() for month in months.values())
    assert np.isclose(total, pax_long.od_pax.sum())
    assert "DFW" in groups["USCA"] and "CUN" in groups["NOTUSCA"]


@pytest.mark.parametrize("script,loader", [("covid_impact_analysis", "load_data"), ("eda_business_indicators", "load"), ("macro_price_analysis", "load_aa")])
def test_existing_team_loaders_preserve_namibia_country_code(tmp_path, script, loader):
    import importlib.util
    passengers = tmp_path / "passengers.csv"
    passengers.write_text("ORG_AIRPRT,DES_AIRPRT,DEP_YEAR,DEP_MONTH,OD_PAX,OD_REV_USD\nABC,DEF,2023,1,10,1000\n")
    airports = tmp_path / "airports.csv"
    airports.write_text("AIRPRT_CD,CNTRY_CD,STATE_PROVNC_CD\nABC,US,TX\nDEF,NA,\n")
    spec = importlib.util.spec_from_file_location(script, config.ROOT / "useful_functions" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, reference = getattr(module, loader)(passengers, airports)
    assert reference.loc[reference.AIRPRT_CD == "DEF", "CNTRY_CD"].iloc[0] == "NA"
