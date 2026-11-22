"""Calendar features that are known in advance for any target month."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def month_features(months) -> pd.DataFrame:
    """Calendar descriptors for each month start in ``months``."""
    months = pd.DatetimeIndex(months)
    days = months.days_in_month.to_numpy()
    weekend = np.empty(len(months), dtype=int)
    for i, m in enumerate(months):
        rng = pd.date_range(m, periods=days[i], freq="D")
        weekend[i] = int((rng.day_of_week >= 5).sum())
    easter_month = np.array([easter(m.year).month == m.month for m in months], dtype=int)
    return pd.DataFrame(
        {
            "cal_month": months.month.to_numpy(),
            "cal_days": days,
            "cal_weekend_days": weekend,
            "cal_easter": easter_month,
            "cal_month_sin": np.sin(2 * np.pi * months.month.to_numpy() / 12),
            "cal_month_cos": np.cos(2 * np.pi * months.month.to_numpy() / 12),
        },
        index=months,
    )
