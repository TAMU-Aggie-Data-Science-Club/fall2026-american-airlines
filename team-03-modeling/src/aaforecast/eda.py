"""Exploratory analysis: figures and a short written summary in outputs/eda."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats_tests
from .config import AA_HUBS
from .data import Panel
from .geo import market_attributes
from .report import md_table


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_eda(panel: Panel, airports: pd.DataFrame, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = panel.months
    attrs = market_attributes(panel.markets, airports)
    dom = attrs["domestic"].to_numpy() == 1
    known = attrs["domestic"].to_numpy() >= 0
    segments = (("Domestic", dom), ("International", known & ~dom), ("Unknown geography", ~known))
    segments = [(label, mask) for label, mask in segments if mask.any()]
    net = pd.Series(panel.pax.sum(0), index=m)
    rev = pd.Series(panel.rev.sum(0), index=m)
    lines = ["# Exploratory Analysis", ""]

    # 1. Network demand and recovery.
    fig, ax = plt.subplots(figsize=(10, 4))
    base = pd.Series(panel.pax[:, m.year == 2019].sum(0), index=range(1, 13))
    for label, mask in segments:
        s = pd.Series(panel.pax[mask].sum(0), index=m)
        b = pd.Series(panel.pax[mask][:, m.year == 2019].sum(0), index=range(1, 13))
        ax.plot(m, s.to_numpy() / b.reindex(m.month).to_numpy() * 100, label=label)
    ax.axhline(100, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("Index, same month of 2019 = 100")
    ax.set_title("Passenger recovery by segment")
    ax.legend()
    _save(fig, out_dir / "recovery_index.png")
    last12 = net.iloc[-12:].sum() / base.sum()
    lines.append(f"* Trailing 12 months are at {last12:.1%} of 2019 passenger volume across the network.")

    # 2. Seasonality profile.
    share = net.groupby([m.year, m.month]).sum().unstack()
    share = share.div(share.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(share.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(share)), share.index)
    ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_title("Monthly share of each year's passengers")
    fig.colorbar(im)
    _save(fig, out_dir / "seasonality_heatmap.png")

    # 3. Average fare trend.
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, mask in segments:
        f = panel.rev[mask].sum(0) / np.maximum(panel.pax[mask].sum(0), 1)
        ax.plot(m, f, label=label)
    ax.set_ylabel("Revenue per passenger (USD)")
    ax.set_title("Average fare by segment")
    ax.legend()
    _save(fig, out_dir / "average_fare.png")

    # 4. Market-level growth dispersion, last 12 months vs prior 12.
    ttm = panel.pax[:, -12:].sum(1)
    prev = panel.pax[:, -24:-12].sum(1)
    ok = (prev > 1000) & (ttm > 0)
    g = np.log(ttm[ok] / prev[ok])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.clip(g, -1, 1), bins=80, weights=ttm[ok] / ttm[ok].sum())
    ax.set_xlabel("Log growth, trailing 12 months vs prior 12")
    ax.set_ylabel("Share of passengers")
    ax.set_title("Markets do not move together")
    _save(fig, out_dir / "market_growth_distribution.png")
    lines.append(f"* Unweighted spread across eligible markets (10th to 90th pct): "
                 f"{np.percentile(g, 10):+.1%} to {np.percentile(g, 90):+.1%} in log terms.")

    # 5. Hubs.
    hub = []
    for h in sorted(AA_HUBS):
        sel = ((panel.markets["org"] == h) | (panel.markets["des"] == h)).to_numpy()
        hub.append({"hub": h, "ttm": panel.pax[sel, -12:].sum(), "prior": panel.pax[sel, -24:-12].sum()})
    hub = pd.DataFrame(hub)
    hub["growth"] = hub["ttm"] / hub["prior"] - 1
    fig, ax = plt.subplots(figsize=(8, 4))
    hub = hub.sort_values("growth")
    ax.barh(hub["hub"], hub["growth"] * 100)
    ax.set_xlabel("Local passenger growth, trailing 12 months (%)")
    ax.set_title("AA hub markets")
    _save(fig, out_dir / "hub_growth.png")

    # 6. Statistical diagnostics.
    post = net[net.index >= "2022-01-01"]
    st = pd.DataFrame(stats_tests.stationarity(post, "Network passengers since 2022"))
    chow = stats_tests.chow_test(net, "2020-03-01")
    top = np.argsort(-panel.pax[:, -12:].sum(1))[:200]
    seas = stats_tests.market_seasonality(panel.pax, m, top)
    tr_net, se_net = stats_tests.seasonal_strength(post)
    st.to_csv(out_dir / "stationarity_tests.csv", index=False)
    seas.to_csv(out_dir / "market_seasonality.csv", index=False)
    lines += [
        f"* Chow test for a structural break at 2020-03: F = {chow['F']:.1f}, p = {chow['p_value']:.2g}. "
        "Pandemic months are excluded from training targets.",
        f"* Network seasonal strength since 2022 is {se_net:.2f} (0 to 1). "
        f"Median across the top 200 markets is {seas['seasonal_strength'].median():.2f}, "
        "so each market keeps its own seasonal anchor.",
        "* Stationarity (ADF and KPSS):",
        "",
        md_table(st, {"adf_p": ".3f", "kpss_p": ".3f"}),
        "",
        f"* Revenue per passenger over the trailing 12 months: ${rev.iloc[-12:].sum() / net.iloc[-12:].sum():,.0f}.",
    ]
    text = "\n".join(lines) + "\n"
    (out_dir / "eda_summary.md").write_text(text, encoding="utf-8")
    return text
