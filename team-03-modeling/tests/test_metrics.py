import numpy as np
import pandas as pd

from aaforecast import metrics


def test_wape_and_bias():
    y = np.array([100.0, 300.0])
    f = np.array([110.0, 270.0])
    assert np.isclose(metrics.wape(y, f), 40 / 400)
    assert np.isclose(metrics.bias(y, f), -20 / 400)


def test_paired_test_detects_better_model():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.uniform(100, 1000, n)
    df = pd.DataFrame({
        "market_id": np.arange(n),
        "y": y,
        "good": y * rng.normal(1, 0.02, n),
        "bad": y * rng.normal(1, 0.2, n),
    })
    res = metrics.paired_test(df, "good", "bad", n_boot=200)
    assert res["wape_gap"] < 0
    assert res["gap_ci_high"] < 0
    assert res["wilcoxon_p"] < 0.01
