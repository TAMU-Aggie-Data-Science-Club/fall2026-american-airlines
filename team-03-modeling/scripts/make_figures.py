"""Build local figures from explicit backtest and forecast run directories.

Run after `python -m aaforecast all`:

    python scripts/make_figures.py --backtest-dir <run> --forecast-dir <run>

Every figure shows relative values only (error rates, coverage, percent
changes, shares). Absolute passenger and revenue counts stay in outputs/,
which is gitignored. Relative metrics and figures also require review before release.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BT = None
FC = None
OUT = None

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED = "#e34948"
BLUE_LIGHT = "#86b6ef"

METHODS = [
    ("anchor_adjust", "Anchor and Adjust", BLUE),
    ("seasonal_anchor", "Seasonal anchor only", ORANGE),
    ("same_month_last_year", "Same month last year", AQUA),
    ("same_month_x_trend", "Last year x trend", YELLOW),
]

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK2,
    }
)

pct0 = PercentFormatter(1.0, decimals=0)
pct1 = PercentFormatter(1.0, decimals=1)
signed_pct2 = FuncFormatter(lambda v, _: f"{v:+.2%}")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def grouped_bars(ax, groups, table, label_first=True):
    n = len(METHODS)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (key, name, color) in enumerate(METHODS):
        vals = table[f"wape_{key}"].to_numpy()
        bars = ax.bar(
            x + (i - (n - 1) / 2) * width,
            vals,
            width * 0.88,
            color=color,
            label=name,
            edgecolor=SURFACE,
            linewidth=1,
        )
        if label_first and i == 0:
            ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=3,
                         fontsize=8.5, color=INK)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(pct0)


def accuracy_by_level() -> None:
    rows = [
        ("Market (O&D)", pd.read_csv(BT / "overall.csv")),
        ("Origin airport", pd.read_csv(BT / "airport_level.csv")),
        ("Network", pd.read_csv(BT / "network_level.csv")),
    ]
    table = pd.concat([r[1] for r in rows], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    grouped_bars(ax, [r[0] for r in rows], table)
    ax.set_ylabel("WAPE, lower is better")
    ax.set_title("Backtest error by planning level")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, "accuracy_by_level.png")


def accuracy_by_horizon() -> None:
    df = pd.read_csv(BT / "by_horizon.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, name, color in METHODS:
        ax.plot(df["horizon"], df[f"wape_{key}"], color=color, lw=2, marker="o",
                ms=5, mec=SURFACE, mew=1.5, label=name)
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("Market WAPE")
    ax.yaxis.set_major_formatter(pct0)
    ax.set_xticks(df["horizon"])
    ax.set_title("Market error by forecast horizon")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save(fig, "accuracy_by_horizon.png")


def accuracy_by_size_and_segment() -> None:
    size = pd.read_csv(BT / "by_size.csv")
    seg = pd.read_csv(BT / "by_segment.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                 gridspec_kw={"width_ratios": [5, 2]}, sharey=True)
    grouped_bars(a1, size["size_band"].tolist(), size)
    a1.set_xlabel("Trailing 12-month passengers per market")
    a1.set_ylabel("Market WAPE")
    a1.set_title("Error by market size")
    grouped_bars(a2, seg["segment"].tolist(), seg)
    a2.set_title("Error by segment")
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.2, wspace=0.08)
    save(fig, "accuracy_by_size_segment.png")


def significance() -> None:
    df = pd.read_csv(BT / "significance.csv")
    names = {k: n for k, n, _ in METHODS}
    df["name"] = df["reference"].map(names)
    df = df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    y = np.arange(len(df))
    ax.hlines(y, df["gap_ci_low"], df["gap_ci_high"], color=BLUE, lw=6, alpha=0.35)
    ax.plot(df["wape_gap"], y, "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [
        f"vs. {r.name}\n{r.wape_gap * 100:+.2f} pts, {r.share_markets_improved:.0%} of markets better"
        for r in df.itertuples()
    ])
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.set_xlim(min(-0.005, df["gap_ci_low"].min() * 1.15), max(0.005, df["gap_ci_high"].max() * 1.15))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v * 100:+.0f} pts"))
    ax.set_xlabel("Change in market WAPE vs. the baseline, with 95% bootstrap interval")
    ax.grid(axis="y", visible=False)
    ax.set_title("Exploratory paired comparison (market bootstrap)")
    save(fig, "significance.png")


def shrink_calibration() -> None:
    df = pd.read_csv(BT / "calibration.csv")
    pooled = df.groupby("shrink")["wape"].mean()
    best = pooled.idxmin()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for _, g in df.groupby("fold"):
        ax.plot(g["shrink"], g["wape"], color=GRID, lw=1.5, zorder=1)
    ax.plot(pooled.index, pooled.values, color=BLUE, lw=2, marker="o", ms=6,
            mec=SURFACE, mew=1.5, zorder=3, label="Mean over folds")
    ax.annotate(f"Retrospective mean minimum: {best:.2f}", (best, pooled[best]), xytext=(0, -18),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.plot([], [], color=GRID, lw=1.5, label="Single fold")
    ax.set_xticks(pooled.index, ["0\nanchor only", "0.25", "0.50", "0.75", "1.0\nfull model"])
    ax.set_xlabel("Weight on the learned market mix")
    ax.set_ylabel("Market WAPE")
    ax.yaxis.set_major_formatter(pct0)
    selected = df.loc[df["selected_for_forecast"], "shrink"].iloc[0]
    ax.axvline(selected, color=ORANGE, ls="--", label=f"Forecast weight from matured evidence: {selected:.2f}")
    ax.set_title("Weight diagnostics; selection uses matured evidence")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    save(fig, "shrink_calibration.png")


def interval_coverage() -> None:
    df = pd.read_csv(BT / "interval_coverage.csv")
    agg = pd.read_csv(BT / "aggregate_interval_table.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8),
                                 gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(df))
    w = 0.38
    a1.bar(x - w / 2, df["coverage"], w * 0.9, color=BLUE, label="Share of markets",
           edgecolor=SURFACE)
    a1.bar(x + w / 2, df["volume_weighted_coverage"], w * 0.9, color=ORANGE,
           label="Volume weighted", edgecolor=SURFACE)
    a1.axhline(0.8, color=INK2, lw=1, ls=(0, (4, 3)))
    a1.annotate("80% target", (len(df) - 0.5, 0.8), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    a1.set_xticks(x, df["fold"])
    a1.set_ylim(0, 1)
    a1.yaxis.set_major_formatter(pct0)
    a1.set_xlabel("Forecast origin (held out)")
    a1.set_title("Coverage on cases with eligible calibration")
    a1.grid(axis="x", visible=False)
    a1.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    y = np.arange(len(agg))[::-1]
    labels = {"airport": "Origin airport", "hub": "AA hub", "network": "Network"}
    a2.hlines(y, agg["q_low"], agg["q_high"], color=BLUE, lw=8, alpha=0.45)
    a2.axvline(0, color=AXIS, lw=1)
    for yi, row in zip(y, agg.itertuples()):
        a2.annotate(f"{row.q_low:+.0%}", (row.q_low, yi), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK2)
        a2.annotate(f"{row.q_high:+.0%}", (row.q_high, yi), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5, color=INK2)
    a2.set_yticks(y, [labels[v] for v in agg["level"]])
    a2.margins(x=0.25)
    a2.xaxis.set_major_formatter(pct0)
    a2.set_xlabel("Log-error quantiles (percent log points)")
    a2.grid(axis="y", visible=False)
    a2.set_title("Bands for rolled-up totals")
    fig.subplots_adjust(wspace=0.35)
    save(fig, "interval_coverage.png")


def hub_accuracy() -> None:
    df = pd.read_csv(BT / "hub_level.csv").sort_values("wape_anchor_adjust", ascending=False)
    y = np.arange(len(df))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 2]})
    a1.hlines(y, df["wape_anchor_adjust"], df["wape_same_month_last_year"], color=GRID, lw=2)
    a1.plot(df["wape_same_month_last_year"], y, "o", color=AQUA, ms=8, mec=SURFACE,
            mew=1.5, label="Same month last year")
    a1.plot(df["wape_anchor_adjust"], y, "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5,
            label="Anchor and Adjust")
    a1.set_yticks(y, df["hub"])
    a1.xaxis.set_major_formatter(pct0)
    a1.set_xlabel("Hub WAPE")
    a1.set_title("Error at each AA hub")
    a1.grid(axis="y", visible=False)
    a1.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    bias = df["bias_anchor_adjust"]
    a2.barh(y, bias, 0.6, color=np.where(bias >= 0, BLUE, RED), edgecolor=SURFACE)
    a2.axvline(0, color=AXIS, lw=1)
    for yi, b in zip(y, bias):
        a2.annotate(f"{b:+.1%}", (b, yi), xytext=(4 if b >= 0 else -4, 0),
                    textcoords="offset points", ha="left" if b >= 0 else "right",
                    va="center", fontsize=8.5, color=INK2)
    a2.xaxis.set_major_formatter(pct0)
    a2.margins(x=0.25)
    a2.set_xlabel("Bias, forecast over actual")
    a2.set_title("Anchor and Adjust bias")
    a2.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "hub_accuracy.png")


def driver_bridge() -> None:
    df = pd.read_csv(FC / "network_bridge.csv")
    steps = df["effect_log_points"].to_numpy()
    left = np.concatenate([[0], np.cumsum(steps)[:-1]])
    right = np.cumsum(steps)
    n = len(df)
    y = np.arange(n + 1)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (a, b) in enumerate(zip(left, right)):
        ax.barh(y[i], b - a, 0.62, left=a, color=BLUE if b >= a else RED, edgecolor=SURFACE)
        pct = df["effect_pct"].iloc[i]
        ax.annotate(f"{pct:+.1%}", (max(a, b), y[i]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
        if i < n - 1:
            ax.plot([b, b], [y[i] - 0.31, y[i + 1] + 0.31], color=AXIS, lw=1)
    total = right[-1]
    ax.barh(y[n], total, 0.62, color=INK2, edgecolor=SURFACE)
    ax.annotate(f"{np.expm1(total):+.1%}", (max(total, 0), y[n]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK, weight="semibold")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, df["driver"].tolist() + ["Forecast vs. anchor"])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}"))
    ax.set_xlabel("Log points vs. seasonal anchor (labels in percent)")
    ax.set_xlim(min(right.min(), 0) - 0.006, max(right.max(), 0) + 0.008)
    ax.grid(axis="y", visible=False)
    ax.set_title("Top-market driver bridge: forecast versus anchor")
    save(fig, "driver_bridge.png")


def feature_importance() -> None:
    df = pd.read_csv(FC / "feature_importance.csv")
    grp = df.groupby("group")["importance"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(grp.index, grp.values, 0.62, color=BLUE, edgecolor=SURFACE)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in grp.values], padding=4,
                 fontsize=8.5, color=INK2)
    ax.set_xlim(min(0, grp.min() * 1.15), max(0.001, grp.max() * 1.15))
    ax.set_xlabel("Model importance, summed over the features in each family")
    ax.grid(axis="y", visible=False)
    ax.set_title("Training-sample permutation importance (diagnostic)")
    save(fig, "feature_importance.png")


def network_outlook() -> None:
    df = pd.read_csv(FC / "network_forecast.csv", parse_dates=["target_month"])
    base = df["last_same_month"]
    mid = df["forecast"] / base - 1
    lo = df["lower"] / base - 1
    hi = df["upper"] / base - 1
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.fill_between(x, lo, hi, color=BLUE_LIGHT, alpha=0.35, lw=0, label="80% interval")
    ax.plot(x, mid, color=BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
            label="Forecast")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x, df["target_month"].dt.strftime("%b\n%Y"))
    ax.yaxis.set_major_formatter(pct0)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.set_ylabel("vs. same month one year earlier")
    ax.set_title(f"Network outlook, {df.target_month.min():%Y-%m} to {df.target_month.max():%Y-%m}")
    ax.legend(loc="lower right")
    save(fig, "network_outlook.png")


def hub_outlook() -> None:
    df = pd.read_csv(FC / "hub_summary.csv")
    base = df["same_months_last_year"]
    df = df.assign(mid=df["forecast"] / base - 1, lo=df["lower"] / base - 1,
                   hi=df["upper"] / base - 1).sort_values("mid")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hlines(y, df["lo"], df["hi"], color=BLUE, lw=6, alpha=0.3, label="80% interval")
    ax.plot(df["mid"], y, "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5, label="Forecast")
    for yi, row in zip(y, df.itertuples()):
        ax.annotate(row.largest_driver, (row.hi, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, df["hub"])
    ax.set_xlim(df["lo"].min() - 0.02, df["hi"].max() + 0.12)
    ax.xaxis.set_major_formatter(pct0)
    ax.set_xlabel("Hub demand vs. same months one year earlier")
    ax.grid(axis="y", visible=False)
    ax.set_title("Hub outlook and the largest driver at each hub")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, "hub_outlook.png")


def scenarios() -> None:
    df = pd.read_csv(FC / "scenarios.csv")
    piv = df.pivot(index="group", columns="scenario", values="change_pct")
    order = ["Domestic", "International"] + sorted(i for i in piv.index if i.startswith("Hub"))
    piv = piv.loc[order]
    lim = np.abs(piv.values).max()
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, f"{v:+.2%}", ha="center", va="center", fontsize=8.5,
                color="#ffffff" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), [g.replace("Hub ", "") for g in piv.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, piv.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, piv.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    ax.axhline(1.5, color=SURFACE, lw=5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, format=signed_pct2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=INK2)
    ax.set_title("Scenario shift in share of network demand (network total held)")
    save(fig, "scenarios.png")


def main() -> None:
    global BT, FC, OUT
    parser = argparse.ArgumentParser(description="Rebuild local figures from explicit completed run directories")
    parser.add_argument("--backtest-dir", type=Path, required=True)
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    BT, FC = args.backtest_dir, args.forecast_dir
    manifests = [json.loads((directory / "manifest.json").read_text()) for directory in (BT, FC)]
    if any(m.get("status") != "complete" for m in manifests):
        raise ValueError("Figures require completed runs")
    if manifests[0]["inputs"] != manifests[1]["inputs"] or manifests[0]["source_sha256"] != manifests[1]["source_sha256"]:
        raise ValueError("Figure inputs use different data or source revisions")
    if [m.get("kind") for m in manifests] != ["backtest", "forecast"]:
        raise ValueError("Expected a backtest run and a forecast run, in that order")
    settings = [{k: v for k, v in m["config"].items() if k != "shrink"} for m in manifests]
    if settings[0] != settings[1]:
        raise ValueError("Figure runs have different scientific configurations")
    OUT = args.out or ROOT / "outputs" / "figures" / (BT.name + "_" + FC.name)
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy_by_level()
    accuracy_by_horizon()
    accuracy_by_size_and_segment()
    significance()
    shrink_calibration()
    interval_coverage()
    hub_accuracy()
    network_outlook()
    optional = [("network_bridge.csv", driver_bridge), ("feature_importance.csv", feature_importance),
                ("hub_summary.csv", hub_outlook), ("scenarios.csv", scenarios)]
    status = {}
    for filename, function in optional:
        path = FC / filename
        if path.exists() and not pd.read_csv(path).empty:
            function()
            status[filename] = "generated"
        else:
            status[filename] = "unavailable_in_this_run"
    (OUT / "optional_figure_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
