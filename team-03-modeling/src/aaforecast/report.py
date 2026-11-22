"""Write planner-facing summaries and charts from backtest and forecast results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "anchor_adjust": "Anchor and Adjust",
    "seasonal_anchor": "Seasonal anchor only",
    "same_month_last_year": "Same month last year",
    "same_month_x_trend": "Last year x trend",
}


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    """Render a small DataFrame as a Markdown table without extra dependencies."""
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in fmt and pd.notna(v):
                cells.append(format(v, fmt[c]))
            elif isinstance(v, float):
                cells.append(f"{v:,.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def backtest_report(res: dict[str, pd.DataFrame], out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(LABELS)

    def level_row(name, df):
        return {"level": name, **{LABELS[m]: df[f"wape_{m}"].iloc[0] for m in methods},
                "bias (A&A)": df["bias_anchor_adjust"].iloc[0]}

    levels = pd.DataFrame([
        level_row("Market (O&D)", res["overall"]),
        level_row("Origin airport", res["airport_level"]),
        level_row("Network", res["network_level"]),
    ])
    hub = res["hub_level"][["hub", "wape_anchor_adjust", "wape_same_month_last_year", "bias_anchor_adjust"]].rename(
        columns={"wape_anchor_adjust": "WAPE A&A", "wape_same_month_last_year": "WAPE last year", "bias_anchor_adjust": "bias A&A"})
    seg = res["by_segment"][["segment", *[f"wape_{m}" for m in methods]]].rename(
        columns={f"wape_{m}": LABELS[m] for m in methods})
    size = res["by_size"][["size_band", *[f"wape_{m}" for m in methods]]].rename(
        columns={f"wape_{m}": LABELS[m] for m in methods})
    sig = res["significance"].assign(reference=lambda d: d["reference"].map(LABELS)).drop(columns="model")
    cal = res["calibration"].groupby("shrink", as_index=False)["wape"].mean()

    # Accuracy by horizon chart.
    bh = res["by_horizon"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        ax.plot(bh["horizon"], bh[f"wape_{m}"] * 100, marker="o", label=LABELS[m])
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("Market-level WAPE (%)")
    ax.set_title("Backtest error by horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "wape_by_horizon.png", dpi=140)
    plt.close(fig)

    pct = {c: ".1%" for c in levels.columns if c != "level"}
    text = "\n".join([
        "# Backtest Summary",
        "",
        "Rolling origins: " + ", ".join(sorted(res["predictions"]["fold"].unique())) + ". Each fold trains only on data available at its origin and forecasts horizons " + f"{int(bh['horizon'].min())} to {int(bh['horizon'].max())}" + ". A horizon is scored only when its target month is already in the extract.",
        "",
        "## Error By Planning Level (WAPE, Lower Is Better)",
        md_table(levels, pct),
        "",
        "## Error By Segment",
        md_table(seg, {c: ".1%" for c in seg.columns if c != "segment"}),
        "",
        "## Error By Market Size (Trailing 12 Month Passengers)",
        md_table(size, {c: ".1%" for c in size.columns if c != "size_band"}),
        "",
        "## Error At AA Hubs",
        md_table(hub, {"WAPE A&A": ".1%", "WAPE last year": ".1%", "bias A&A": "+.1%"}),
        "",
        "## Exploratory Paired Comparison",
        md_table(sig, {"wape_gap": "+.2%", "gap_ci_low": "+.2%", "gap_ci_high": "+.2%", "share_markets_improved": ".1%", "wilcoxon_p": ".2g"}),
        "",
        "A negative WAPE gap means Anchor and Adjust misses by less. Bootstrap units are markets; shared airport/time shocks and overlapping windows limit inferential claims. This is retrospective evaluation, not proof of causal effects or an untouched holdout.",
        "",
        "## Weight On The Learned Market Mix",
        md_table(cal, {"shrink": ".2f", "wape": ".2%"}),
        "",
        "Each fold used the weight that scored best on earlier folds only. The forecast uses the best weight overall.",
        "",
        "## Prediction Interval Coverage (Held-Out Folds)",
        md_table(res["interval_coverage"], {"coverage": ".1%", "volume_weighted_coverage": ".1%"}),
        "",
    ])
    (out_dir / "backtest_summary.md").write_text(text, encoding="utf-8")
    return text


def forecast_report(res: dict[str, pd.DataFrame], out_dir: Path, target_label: str = "passengers") -> str:
    if "network_bridge" not in res:
        text = "# Forecast Summary\n\nMinimal run: companion explanations and scenarios were skipped.\n\n" + md_table(res["network_forecast"])
        (out_dir / "forecast_summary.md").write_text(text, encoding="utf-8")
        return text
    out_dir.mkdir(parents=True, exist_ok=True)
    nf = res["network_forecast"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = pd.to_datetime(nf["target_month"])
    ax.bar(x, nf["forecast"], width=20, label="Forecast")
    ax.errorbar(x, nf["forecast"], yerr=[nf["forecast"] - nf["lower"], nf["upper"] - nf["forecast"]], fmt="none", color="k", capsize=3)
    ax.plot(x, nf["last_same_month"], "o", color="tab:orange", label="Same month, last year")
    ax.set_title(f"Network {target_label}, forecast horizon")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "network_forecast.png", dpi=140)
    plt.close(fig)

    nb = res["network_bridge"]
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = nb["effect_pct"].to_numpy() * 100
    ax.barh(nb["driver"], vals, color=np.where(vals >= 0, "tab:green", "tab:red"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Effect on forecast vs anchor (%)")
    ax.set_title("Network driver bridge")
    fig.tight_layout()
    fig.savefig(out_dir / "driver_bridge.png", dpi=140)
    plt.close(fig)

    total_f, total_l = nf["forecast"].sum(), nf["last_same_month"].sum()
    wl = res["watchlist"]
    flags = wl["flag"].value_counts().rename_axis("flag").reset_index(name="markets")
    hub = res["hub_summary"][["hub", "markets", "forecast", "vs_last_year_pct", "vs_anchor_pct", "largest_driver"]]
    top_wl = wl.head(15)[["org", "des", "anchor", "forecast", "vs_anchor_pct", "vs_last_year_pct", "flag"]]
    imp = res["feature_importance"].groupby("group")["importance"].sum().sort_values(ascending=False).reset_index()
    money = {c: ",.0f" for c in ("last_same_month", "anchor", "forecast", "lower", "upper")}
    text = "\n".join([
        "# Forecast Summary",
        "",
        f"Target months {nf['target_month'].min():%Y-%m} to {nf['target_month'].max():%Y-%m}. "
        f"The forecast is {total_f / total_l - 1:+.1%} versus the same months one year earlier.",
        "",
        "## Network By Month",
        md_table(nf.assign(target_month=pd.to_datetime(nf["target_month"]).dt.strftime("%Y-%m")), money),
        "",
        "## Driver Bridge (Volume Weighted, Top-Market Subset)",
        "The bridge covers up to the largest 2,000 markets, not an exact whole-network attribution.",
        md_table(nb, {"effect_log_points": "+.3f", "effect_pct": "+.1%"}),
        "",
        "## AA Hubs",
        md_table(hub, {"forecast": ",.0f", "vs_last_year_pct": "+.1%", "vs_anchor_pct": "+.1%"}),
        "",
        "## Watchlist",
        md_table(flags),
        "",
        md_table(top_wl, {"anchor": ",.0f", "forecast": ",.0f", "vs_anchor_pct": "+.1%", "vs_last_year_pct": "+.1%"}),
        "",
        "## Scenarios (Shift In Share, Network Total Held)",
        md_table(res["scenarios"].pivot(index="group", columns="scenario", values="change_pct").reset_index(),
                 {s: "+.2%" for s in res["scenarios"]["scenario"].unique()}),
        "",
        "Scenario effects are sensitivities learned from history, not causal estimates.",
        "",
        "## What The Model Leans On",
        md_table(imp, {"importance": ".4f"}),
        "",
    ])
    (out_dir / "forecast_summary.md").write_text(text, encoding="utf-8")
    return text
