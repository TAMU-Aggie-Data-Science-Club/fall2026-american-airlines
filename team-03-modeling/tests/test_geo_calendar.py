import datetime as dt

import numpy as np
import pandas as pd

from aaforecast import calendar, geo
from aaforecast.external import catchment


def test_haversine_dfw_lax():
    d = geo.haversine_km(32.8998, -97.0403, 33.9416, -118.4085)
    assert 1960 < d < 2000


def test_nearest_respects_distance_cap():
    idx, dist = geo.nearest([32.9, 10.0], [-97.0, 10.0], [32.85, 41.98], [-96.85, -87.9], max_km=200)
    assert idx[0] == 0 and idx[1] == -1
    assert dist[0] < 20


def test_market_attributes(airports):
    markets = pd.DataFrame({"org": ["DFW", "DFW"], "des": ["LAX", "CUN"]})
    a = geo.market_attributes(markets, airports)
    assert a["domestic"].tolist() == [1, 0]
    assert a["foreign_country"].tolist() == ["", "MX"]
    assert a["org_hub"].tolist() == [1, 1]


def test_easter_known_dates():
    assert calendar.easter(2024) == dt.date(2024, 3, 31)
    assert calendar.easter(2025) == dt.date(2025, 4, 20)
    assert calendar.easter(2026) == dt.date(2026, 4, 5)


def test_month_features():
    f = calendar.month_features(pd.date_range("2024-03-01", periods=2, freq="MS"))
    assert f["cal_easter"].tolist() == [1, 0]
    assert f["cal_days"].tolist() == [31, 30]
    assert f["cal_weekend_days"].iloc[0] == 10  # March 2024 had five Saturdays and five Sundays


def test_catchments_pool_metro_airports(airports):
    points = pd.DataFrame({"fips": ["48113", "48439", "06037"], "lat": [32.77, 32.77, 34.05], "lon": [-96.78, -97.29, -118.24]})
    assign = catchment.assign_counties(points, airports)
    assert set(assign["metro"]) == {"DFW", "LAX"}
    pop = pd.DataFrame({"fips": ["48113", "48439", "06037"], "year": 2024, "population": [100, 50, 300]})
    ap = catchment.airport_population(assign, pop, airports).set_index("airport")["catchment_pop"]
    assert ap["DFW"] == ap["DAL"] == 150
    assert ap["LAX"] == 300
    assert np.isnan(ap.get("CUN", np.nan))
