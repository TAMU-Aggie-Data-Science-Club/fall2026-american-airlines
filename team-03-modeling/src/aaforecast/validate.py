"""Data quality checks on the raw and cleaned AA datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import PAX_COLUMNS


@dataclass
class Check:
    name: str
    status: str  # PASS, WARN, FAIL
    detail: str


def _status(ok: bool, warn_only: bool = False) -> str:
    if ok:
        return "PASS"
    return "WARN" if warn_only else "FAIL"


def check_raw(raw: pd.DataFrame) -> list[Check]:
    """Checks that run on the renamed raw table before cleaning."""
    out: list[Check] = []
    expected = set(PAX_COLUMNS.values())
    missing = expected - set(raw.columns)
    out.append(Check("required columns", _status(not missing), f"missing: {sorted(missing)}" if missing else "all present"))
    nulls = raw[list(expected & set(raw.columns))].isna().sum()
    out.append(Check("null values", _status(nulls.sum() == 0, True), f"{int(nulls.sum())} nulls"))
    keys = ["year", "month_num", "org", "des"]
    dups = int(raw.duplicated(keys).sum())
    out.append(Check("unique market-month keys", _status(dups == 0, True), f"{dups} duplicate keys"))
    for col in ("od_pax", "od_rev_usd"):
        neg = int((raw[col] < 0).sum())
        out.append(Check(f"non-negative {col}", _status(neg == 0), f"{neg} negative rows"))
    self_od = int((raw["org"] == raw["des"]).sum())
    out.append(Check("origin differs from destination", _status(self_od == 0, True), f"{self_od} rows"))
    bad_month = int((~raw["month_num"].between(1, 12)).sum())
    out.append(Check("month in 1..12", _status(bad_month == 0), f"{bad_month} rows"))
    loads = raw["load_ts"].nunique()
    out.append(Check("single load snapshot", _status(loads == 1, True), f"{loads} load timestamps"))
    return out


def check_clean(df: pd.DataFrame, airports: pd.DataFrame) -> list[Check]:
    """Checks on the cleaned long table and its join to the airport reference."""
    out: list[Check] = []
    months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    gaps = sorted(set(months) - set(df["month"].unique()))
    out.append(
        Check(
            "contiguous months",
            _status(not gaps),
            f"{df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}, {len(gaps)} empty months",
        )
    )
    codes = set(df["org"]) | set(df["des"])
    unknown = sorted(codes - set(airports.index))
    out.append(Check("airports found in reference", _status(not unknown), f"unknown: {unknown[:10]}" if unknown else f"{len(codes)} airports matched"))

    per_month = df.groupby("month").size()
    last_ratio = per_month.iloc[-1] / per_month.iloc[-13:-1].median()
    out.append(
        Check(
            "latest month completeness",
            _status(last_ratio > 0.95, True),
            f"latest month has {last_ratio:.1%} of the trailing median row count",
        )
    )

    n_months = len(months)
    coverage = df.groupby(["org", "des"]).size() / n_months
    out.append(
        Check(
            "market history coverage",
            _status(coverage.min() > 0.5, True),
            f"{(coverage == 1).mean():.1%} of markets report every month; min coverage {coverage.min():.1%}",
        )
    )

    fare = df["od_rev_usd"] / df["od_pax"].where(df["od_pax"] > 0)
    odd_fare = int(((fare < 5) | (fare > 20000)).sum())
    out.append(Check("average fare within $5 to $20k", _status(odd_fare == 0, True), f"{odd_fare} rows outside range"))

    zero_pax_rev = int(((df["od_pax"] == 0) & (df["od_rev_usd"] > 0)).sum())
    out.append(Check("revenue only with passengers", _status(zero_pax_rev == 0, True), f"{zero_pax_rev} rows"))

    # Robust outliers: same-month year-over-year log changes far from the market median.
    s = df.set_index(["org", "des", "month"])["od_pax"]
    # Calendar-month join, not 12 observed rows (which is wrong for sparse markets).
    prior = df[["org", "des", "month", "od_pax"]].copy()
    prior["month"] = prior["month"] + pd.DateOffset(months=12)
    lag = prior.set_index(["org", "des", "month"])["od_pax"].reindex(s.index)
    yoy = np.log1p(s) - np.log1p(lag)
    yoy = yoy[(s.index.get_level_values("month") >= "2022-01-01") & lag.notna()]
    med = yoy.median()
    mad = (yoy - med).abs().median() * 1.4826
    extreme = int(((yoy - med).abs() > 6 * mad).sum()) if mad > 0 else 0
    out.append(
        Check(
            "year-over-year outliers since 2022",
            _status(extreme / max(len(yoy), 1) < 0.01, True),
            f"{extreme} market-months beyond 6 robust SD ({extreme / max(len(yoy), 1):.2%})",
        )
    )
    return out


def to_markdown(checks: list[Check], title: str = "Data Quality Report") -> str:
    lines = [f"# {title}", "", "| Check | Status | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c.name} | {c.status} | {c.detail} |" for c in checks]
    return "\n".join(lines) + "\n"


def to_frame(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in checks])
