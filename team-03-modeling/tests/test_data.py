import numpy as np
import pandas as pd

from aaforecast import data, validate


def test_load_pax_never_deserializes_legacy_pickle(monkeypatch, tmp_path, raw_pax):
    monkeypatch.setattr(pd, "read_pickle", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Legacy executable cache must never be loaded")))
    monkeypatch.setattr(data, "read_pax_raw", lambda path: raw_pax)
    expected = data.clean_pax(raw_pax)
    pd.testing.assert_frame_equal(data.load_pax(tmp_path / "source.csv"), expected)


def test_clean_sums_duplicates_and_drops_bad_rows(raw_pax):
    dup = raw_pax.iloc[[0]].copy()
    neg = raw_pax.iloc[[1]].assign(od_pax=-5.0)
    self_od = raw_pax.iloc[[2]].assign(des=raw_pax.iloc[2]["org"])
    raw = pd.concat([raw_pax.iloc[3:], dup, raw_pax.iloc[[0]], neg, self_od], ignore_index=True)
    clean = data.clean_pax(raw)
    assert not clean.duplicated(["month", "org", "des"]).any()
    assert (clean["od_pax"] >= 0).all()
    assert (clean["org"] != clean["des"]).all()
    first = raw_pax.iloc[0]
    row = clean[(clean["org"] == first["org"]) & (clean["des"] == first["des"]) & (clean["month"] == "2019-01-01")]
    assert np.isclose(row["od_pax"].iloc[0], 2 * first["od_pax"])


def test_dms_to_decimal_handles_hemispheres():
    out = data.dms_to_decimal([32, 33], [54, 56], [0, 24], ["N", "S"])
    assert np.allclose(out, [32.9, -33.94])
    assert np.isclose(data.dms_to_decimal([97], [2], [24], ["W"])[0], -97.04)


def test_build_panel_is_dense(pax_long):
    sparse = pax_long.drop(index=pax_long.index[::7])
    panel = data.build_panel(sparse)
    assert panel.pax.shape == (30, 60)
    assert panel.observed.sum() == len(sparse)
    assert (panel.pax[~panel.observed] == 0).all()


def test_validation_flags_negative_values(raw_pax):
    bad = raw_pax.copy()
    bad.loc[0, "od_pax"] = -1
    checks = {c.name: c.status for c in validate.check_raw(bad)}
    assert checks["non-negative od_pax"] == "FAIL"
    assert checks["required columns"] == "PASS"
