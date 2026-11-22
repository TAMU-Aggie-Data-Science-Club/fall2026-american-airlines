"""The most important property: nothing after the origin month leaks into features."""

import numpy as np
import pandas as pd
import pytest

from aaforecast import data
from aaforecast.backtest import aggregate_intervals, apply_aggregate_interval, apply_intervals, interval_table
from aaforecast.config import RunConfig
from aaforecast.features import FeatureBuilder, anchor_index, feature_columns, training_origins
from aaforecast.geo import market_attributes
from aaforecast.model import AnchorAdjustModel, baselines, network_scale, relative_adjustment

SMALL_GBM = {"max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 8, "min_samples_leaf": 5, "early_stopping": False}


def _builder(pax_long, airports, cfg=None):
    panel = data.build_panel(pax_long)
    cfg = cfg or RunConfig(min_ttm=0.0, gbm_params=SMALL_GBM)
    return FeatureBuilder(panel, market_attributes(panel.markets, airports), None, cfg, airports)


def test_anchor_index():
    assert anchor_index(40, 12) == 40
    assert anchor_index(40, 13) == 29
    assert anchor_index(40, 18) == 34
    for h in range(12, 19):
        assert (40 + h - anchor_index(40, h)) % 12 == 0


def test_training_origins_end_at_last_month():
    o = training_origins(60, 3)
    assert o[-1] == 59 and all(b - a == 3 for a, b in zip(o, o[1:])) and o[0] >= 11


def test_features_do_not_see_the_future(pax_long, airports):
    t = 40
    cutoff = data.build_panel(pax_long).months[t]
    before = _builder(pax_long, airports).origin_rows(t)
    changed = pax_long.copy()
    future = changed["month"] > cutoff
    changed.loc[future, ["od_pax", "od_rev_usd"]] *= 7.0
    after = _builder(changed, airports).origin_rows(t)
    cols = feature_columns(before) + ["anchor", "last_same_month", "ttm"]
    pd.testing.assert_frame_equal(before[cols], after[cols])
    assert not np.allclose(before["y"], after["y"])  # only the target moved


def test_relative_target_is_centered(pax_long, airports):
    rows = _builder(pax_long, airports).build([30, 33])
    rows = rows[rows["y"].notna()]
    z = relative_adjustment(rows)
    w = rows["anchor"].clip(lower=1)
    for _, g in rows.assign(z=z, w=w).groupby(["origin_idx", "horizon"]):
        assert abs((g["z"] * g["w"]).sum() / g["w"].sum()) < 1e-9


def test_network_scale_hits_outlook(pax_long, airports):
    rows = _builder(pax_long, airports).origin_rows(40)
    noisy = rows["anchor"].to_numpy() * np.exp(np.random.default_rng(0).normal(0, 0.3, len(rows)))
    scaled = network_scale(rows, noisy, growth=0.05)
    for h, g in rows.assign(f=scaled).groupby("horizon"):
        assert np.isclose(g["f"].sum(), g["anchor"].sum() * 1.05 ** (h / 12))


def test_model_fits_and_predicts(pax_long, airports):
    cfg = RunConfig(min_ttm=0.0, gbm_params=SMALL_GBM)
    b = _builder(pax_long, airports, cfg)
    rows = b.build(training_origins(len(b.p.months), 3))
    model = AnchorAdjustModel(cfg).fit(rows[rows["target_idx"] <= 47])
    test = b.origin_rows(47)
    f = model.predict(test)
    assert np.all(np.isfinite(f)) and np.all(f >= 0)
    assert set(baselines(test)) == {"seasonal_anchor", "same_month_last_year", "same_month_x_trend"}


def test_intervals_from_backtest_errors():
    rng = np.random.default_rng(0)
    n = 2000
    pred = pd.DataFrame({
        "fold": rng.choice(["a", "b", "c"], n),
        "horizon": rng.integers(12, 19, n),
        "ttm": rng.uniform(1_000, 300_000, n),
        "anchor_adjust": rng.uniform(100, 10_000, n),
    })
    pred["y"] = pred["anchor_adjust"] * np.exp(rng.normal(0, 0.1, n))
    pred["origin"] = pd.to_datetime(pred["fold"].map({"a": "2018-01-01", "b": "2021-01-01", "c": "2024-01-01"}))
    pred["target_month"] = pred["origin"] + pd.DateOffset(months=18)
    table, cov = interval_table(pred, 0.8)
    assert cov.loc[cov["status"] == "evaluated", "coverage"].between(0.7, 0.9).all()
    assert cov.set_index("fold").loc["a", "status"] == "unavailable"
    out = apply_intervals(pred.rename(columns={"anchor_adjust": "forecast"}), table)
    assert (out["lower"] <= out["forecast"]).all() and (out["upper"] >= out["forecast"]).all()


def test_aggregate_intervals_are_narrower_than_summed_bounds():
    rng = np.random.default_rng(3)
    n_mkt, n_month = 300, 35
    f = rng.uniform(100, 1000, (n_mkt, n_month))
    y = f * np.exp(rng.normal(0, 0.2, (n_mkt, n_month)))  # independent market errors
    net = pd.DataFrame({"y": y.sum(0), "anchor_adjust": f.sum(0)})
    table = aggregate_intervals({"network": net}, 0.8)
    out = apply_aggregate_interval(pd.DataFrame({"forecast": [f.sum(0).mean()]}), table, "network")
    rel_width = (out["upper"] - out["lower"]).iloc[0] / out["forecast"].iloc[0]
    assert rel_width < 0.1  # summing 80% market bounds would give roughly 0.5


@pytest.mark.parametrize("h", [12, 15, 18])
def test_seasonal_anchor_tracks_shape(pax_long, airports, h):
    rows = _builder(pax_long, airports).origin_rows(47, horizons=(h,))
    ratio = rows["anchor"] / rows["last_same_month"]
    assert ratio.between(0.6, 1.6).all()
