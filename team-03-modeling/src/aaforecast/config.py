"""Paths and run settings shared across the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("AAF_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DIR = ROOT / "outputs"

PAX_FILE = Path(os.environ.get("AAF_PAX_FILE", RAW_DIR / "AA_PAX_FORECASTING.csv"))
PUBLIC_AIRPORT_FILE = EXTERNAL_DIR / "public_airports" / "AA_AIRPORT_INFO.csv"
_default_airports = RAW_DIR / "AA_AIRPORT_INFO.csv"
if not _default_airports.exists() and PUBLIC_AIRPORT_FILE.exists():
    _default_airports = PUBLIC_AIRPORT_FILE
AIRPORT_FILE = Path(os.environ.get("AAF_AIRPORT_FILE", _default_airports))

# American Airlines hubs (public information).
AA_HUBS = frozenset({"DFW", "CLT", "ORD", "MIA", "PHL", "PHX", "LAX", "JFK", "LGA", "DCA"})

# Months whose demand was distorted by the COVID-19 shock. Targets inside this
# window are excluded from training so the model does not learn a collapse and
# rebound that is not expected to repeat.
PANDEMIC_START = "2020-03"
PANDEMIC_END = "2021-12"

TARGETS = {"pax": "od_pax", "rev": "od_rev_usd"}


@dataclass
class RunConfig:
    """Settings for a backtest or forecast run."""

    target: str = "pax"
    horizons: tuple[int, ...] = tuple(range(12, 19))
    origin_step: int = 3
    min_ttm: float = 120.0
    backtest_origins: tuple[str, ...] = ("2024-01", "2024-04", "2024-07", "2024-10", "2025-01")
    interval: float = 0.8
    # Network outlook: annual growth applied to the sum of anchors. 0 means flat.
    network_growth: float = 0.0
    # Weight on the learned market mix (0 = pure anchor). Chosen by the backtest.
    shrink: float = 0.0
    shrink_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    use_external: bool = False
    label_lag_months: int = 1
    minimal_forecast: bool = False
    passenger_only: bool = False
    disabled_groups: tuple[str, ...] = ()
    fast: bool = False
    fast_top_n: int = 1500
    seed: int = 7
    gbm_params: dict = field(
        default_factory=lambda: {
            "max_iter": 300,
            "learning_rate": 0.03,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 1000,
            "l2_regularization": 5.0,
            "early_stopping": False,
        }
    )

    @property
    def target_col(self) -> str:
        return TARGETS[self.target]


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, EXTERNAL_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
