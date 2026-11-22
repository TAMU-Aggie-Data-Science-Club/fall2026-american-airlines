"""Fill team 10's reports and build the final slide deck from the notebooks' outputs, locally.

Run after notebooks 01-05 (04 and 05 write d4_scores.json / d5_scores.json):

    python team-10-modeling/scripts/build_deliverables.py

Nothing leaves the machine. Outputs follow the data mode (see src/data.py):

* REAL mode -> data/interim/team-10/reports/*.md and data/interim/team-10/team10_final_slides.pptx
  (git-ignored). Review them, then copy the reports into team-10-modeling/results/ yourself.
* FAKE mode -> data/fake/build/ (git-ignored), stamped FAKE everywhere.

Only rounded percentages go into the reports. Raw passenger totals are never written.
Needs python-pptx for the deck (pip install python-pptx); the reports are built without it.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

MODELING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELING / "src"))

from data import FIGS, MODE, REPO_ROOT, TABLES  # noqa: E402  (paths only, loads no data)

NAIVE = "Seasonal naive"
TODAY = dt.date.today().isoformat()
OUT = TABLES if MODE == "REAL" else REPO_ROOT / "data" / "fake" / "build"
REPORTS = ["results/model-comparison.md", "results/baseline-decision.md",
           "results/forecast-evaluation.md", "results/final-model-report.md", "README.md"]


def load(name: str) -> dict:
    path = TABLES / name
    if not path.exists():
        sys.exit(f"Missing {path}. Run notebooks 04 and 05 first.")
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x, d=1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def spct(x) -> str:
    return "n/a" if x is None else f"{100 * x:+.1f}%"


def pts(x) -> str:
    return f"{100 * x:+.1f} pts"


# --------------------------------------------------------------------------- content
def build_blocks(d4: dict, d5: dict) -> dict[str, str]:
    carried, test = d5["carried"], d5["test"]
    tc, tn = test[carried], test[NAIVE]
    cov = d5["coverage"]
    names = list(d4["scores"]["select"].keys())
    sel, stab = d4["scores"]["select"], d4["scores"]["stability"]
    others = {n: s for n, s in test.items() if n != carried}
    best_other = min(others, key=lambda n: others[n]["WAPE_24m"])
    b = {}

    # D4 table + read-out
    rows = ["| origin | candidate | WAPE y1 | WAPE y2 | WAPE 24m | bias 24m |", "|---|---|---|---|---|---|"]
    for lab, scores in (("select", sel), ("stability", stab)):
        for n in names:
            s = scores.get(n)
            star = " **(carried)**" if n == d4["carried"] and lab == "select" else ""
            if s is None:
                rows.append(f"| {lab} ({d4['origins'][lab]}) | {n}{star} | not fittable | | | |")
            else:
                rows.append(f"| {lab} ({d4['origins'][lab]}) | {n}{star} | {pct(s['WAPE_y1'])} | "
                            f"{pct(s['WAPE_y2'])} | {pct(s['WAPE_24m'])} | {spct(s['bias_24m'])} |")
    b["d4_table"] = "Route-month WAPE (lower is better), rounded, from the real run on " + TODAY + ".\n\n" + "\n".join(rows)

    fit = {n: s for n, s in sel.items() if s is not None}
    best_y1 = min(fit, key=lambda n: fit[n]["WAPE_y1"])
    best_y2 = min(fit, key=lambda n: fit[n]["WAPE_y2"])
    gap = sel[NAIVE]["WAPE_y2"] - sel[NAIVE]["WAPE_y1"]
    b["d4_readout"] = (
        f"**Read-out.** At the select origin the lowest year-1 WAPE belongs to *{best_y1}* and the lowest "
        f"year-2 WAPE to *{best_y2}*. The seasonal naive's error grows by {pts(gap)} from year 1 to year 2. "
        f"Carry rule results:\n\n" + "\n".join(f"- {n}" for n in d4["notes"])
        + f"\n\n**Carried into D5: {d4['carried']}.**")

    # Decision
    if d4["carried"] == NAIVE:
        margin = "— (the naive is the reference; no challenger passed all three checks)"
        stab_txt = "not applicable (default carried)"
    else:
        m1 = sel[d4["carried"]]["WAPE_y1"] - sel[NAIVE]["WAPE_y1"]
        m2 = sel[d4["carried"]]["WAPE_y2"] - sel[NAIVE]["WAPE_y2"]
        margin = f"{pts(m1)} / {pts(m2)}"
        stab_txt = "pass"
    b["decision"] = (
        "| | |\n|---|---|\n"
        f"| Carried model | **{d4['carried']}** |\n"
        f"| Margin over the naive, select origin, year 1 / year 2 (negative = better) | {margin} |\n"
        f"| Stability check | {stab_txt} |\n"
        f"| Decided by / date | carry rule in `src/forecasting.py`, run {TODAY}; team review: *(add names)* |\n\n"
        "Rule output:\n\n" + "\n".join(f"- {n}" for n in d4["notes"]))

    # D5 table
    label_c = f"{carried} (carried)"
    rows = ["| | year 1 | year 2 | both years |", "|---|---|---|---|",
            f"| {label_c}, route-month WAPE | {pct(tc['WAPE_y1'])} | {pct(tc['WAPE_y2'])} | {pct(tc['WAPE_24m'])} |"]
    if carried != NAIVE:
        rows.append(f"| Seasonal naive, route-month WAPE | {pct(tn['WAPE_y1'])} | {pct(tn['WAPE_y2'])} | {pct(tn['WAPE_24m'])} |")
    else:
        ob = test[best_other]
        rows.append(f"| Best challenger ({best_other}), route-month WAPE, context only | {pct(ob['WAPE_y1'])} | "
                    f"{pct(ob['WAPE_y2'])} | {pct(ob['WAPE_24m'])} |")
    rows += [f"| {carried} bias (forecast ÷ actual − 1) | {spct(tc['bias_y1'])} | {spct(tc['bias_y2'])} | {spct(tc['bias_24m'])} |",
             f"| Network monthly WAPE | | | {pct(tc['network_WAPE'])} |",
             f"| 80% network band coverage | {pct(cov['y1'], 0)} | {pct(cov['y2'], 0)} | {pct(cov['all'], 0)} |"]
    b["d5_table"] = f"Test window from {d5['origins']['test']}, one 24-month forecast, rounded.\n\n" + "\n".join(rows)

    # NDA: the committed reports carry error metrics only. Each segment's share of passengers is
    # business data, so it stays out of the repo (it is still in the local slides).
    seg = ["| segment | level | year | WAPE (carried) | WAPE (naive) |", "|---|---|---|---|---|"]
    seg += [f"| {r['segment']} | {r['level']} | {r['year']} | {pct(r['WAPE'])} | {pct(r['WAPE_naive'])} |"
            for r in d5["segments"]]
    b["d5_segments"] = "\n".join(seg)

    t1 = tc["WAPE_y2"] - tc["WAPE_y1"] <= 0.05
    t2 = carried == NAIVE or tc["WAPE_y2"] <= tn["WAPE_y2"]
    t3 = 0.65 <= cov["all"] <= 0.95
    claim = ("The data supports evaluating a two-year forecast, and on this test the forecast holds up "
             "across both years with a usable band." if (t1 and t2 and t3) else
             "The pipeline produces a 24-month forecast, but at least one test fails, so the two-year "
             "claim must be stated with the caveat(s) above.")
    b["d5_verdict"] = (
        f"1. Year-2 WAPE close to year 1 (within 5 pts): **{'pass' if t1 else 'fail'}** "
        f"({pct(tc['WAPE_y1'])} → {pct(tc['WAPE_y2'])}).\n"
        f"2. Carried model at least matches the naive in year 2: **{'pass' if t2 else 'fail'}**"
        + (" (the carried model *is* the naive)." if carried == NAIVE else f" ({pct(tc['WAPE_y2'])} vs {pct(tn['WAPE_y2'])}).") + "\n"
        f"3. 80% band coverage between 65% and 95%: **{'pass' if t3 else 'fail'}** ({pct(cov['all'], 0)}).\n\n"
        f"**Verdict:** {claim}")

    # Status / summary / figures
    if MODE == "REAL":
        b["status"] = (f"> Status: **run on the real AA data on {TODAY}** (notebooks 01–05). Numbers are rounded; "
                       "unrounded outputs stay in `data/interim/team-10/` (git-ignored).")
        b["readme_status"] = f"**Reports in `results/` were filled from a real-data run on {TODAY}** (rounded numbers only)."
    else:
        b["status"] = "> Status: **FAKE-mode build (pipeline check).** Numbers below come from generated data and say nothing about AA."
        b["readme_status"] = "**FAKE-mode build: numbers are from generated data.**"

    diff = tc["WAPE_24m"] - tn["WAPE_24m"]
    beats = "is about level with" if abs(diff) < 0.005 else ("beats" if diff < 0 else "does not beat")
    vs_naive = ("It is the seasonal naive itself: no challenger cleared the pre-registered bar"
                if carried == NAIVE else f"It {beats} the seasonal naive over the two years "
                f"({pct(tc['WAPE_24m'])} vs {pct(tn['WAPE_24m'])} route-month WAPE)")
    b["summary"] = (
        f"We forecast passengers for every route and month two years ahead, in one shot, with **{carried}**. "
        f"On the held-out test window its typical route-month error is {pct(tc['WAPE_y1'])} in year 1 and "
        f"{pct(tc['WAPE_y2'])} in year 2, with a network-level error of {pct(tc['network_WAPE'])}. {vs_naive}. "
        f"It is most useful as a network-level two-year demand outlook and a year-1 route baseline for large routes; "
        f"its 80% band covered {pct(cov['all'], 0)} of test months.")

    segs = {(r["segment"], r["level"], r["year"]): r for r in d5["segments"]}
    big, small = segs.get(("volume", "Q4 largest", 1)), segs.get(("volume", "Q1 smallest", 1))
    dom, intl = segs.get(("market", "domestic", 1)), segs.get(("market", "international", 1))
    hz = d5["by_horizon"][carried]
    seg_line = (f"year-1 WAPE is {pct(big['WAPE'])} on the largest routes vs {pct(small['WAPE'])} on the smallest"
                if big and small else "see the segment table")
    mk_line = (f"; domestic {pct(dom['WAPE'])} vs international {pct(intl['WAPE'])}" if dom and intl else "")
    # NDA: forward growth and passenger volumes are not written into committed reports.
    b["figures"] = "\n".join([
        f"- `d5_observed_vs_forecast_indexed.png` (committed; indexed, no passenger counts): the network forecast is "
        f"{spct(tc['bias_y1'])} off in total in year 1 and {spct(tc['bias_y2'])} in year 2; the 80% band contains "
        f"{pct(cov['all'], 0)} of test months.",
        f"- `d5_error_by_horizon.png`: monthly WAPE runs from {pct(min(hz))} to {pct(max(hz))}; year-2 error is "
        f"{pts(tc['WAPE_y2'] - tc['WAPE_y1'])} vs year 1."
        + ("" if carried == NAIVE else f" The carried model beats the naive in {d5['months_carried_beats_naive']} of 24 months."),
        f"- `d5_error_by_segment.png`: {seg_line}{mk_line}.",
        f"- `d5_residuals.png` (kept local): median log(actual ÷ forecast) is {d5['residual']['median_log']:+.3f}; "
        f"{pct(d5['residual']['share_under'], 0)} of matched route-months are under-forecast.",
        "- `d5_forward_forecast.png` (kept local, NDA: shows passenger volumes): the forecast for the next 24 months; "
        "shown only in the slides submitted to the PMs.",
        "- `d4_wape_by_horizon.png` (committed): all four candidates at both D4 origins, month by month.",
    ])
    return b


def fill_reports(blocks: dict[str, str]) -> list[Path]:
    out_dir = OUT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rel in REPORTS:
        text = (MODELING / rel).read_text(encoding="utf-8")

        def sub(m):
            key = m.group(1)
            return f"<!-- FILL:{key} -->\n{blocks[key]}\n<!-- /FILL -->" if key in blocks else m.group(0)

        text = re.sub(r"<!-- FILL:(\w+) -->.*?<!-- /FILL -->", sub, text, flags=re.S)
        path = out_dir / Path(rel).name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


# --------------------------------------------------------------------------- slides
def build_deck(d4: dict, d5: dict, blocks: dict[str, str]) -> Path | None:
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.util import Inches, Pt
    except ImportError:
        print("python-pptx not installed: skipping the deck (pip install python-pptx).")
        return None
    from PIL import Image

    NAVY, GREY, RED = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0x55, 0x5F, 0x6B), RGBColor(0xB0, 0x24, 0x18)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    count = [0]

    def text(slide, s, left, top, width, height, size=18, bold=False, color=GREY):
        tf = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)).text_frame
        tf.word_wrap = True
        lines = s if isinstance(s, list) else [s]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
            p.space_after = Pt(8)
        return tf

    def slide(title, notes=""):
        count[0] += 1
        s = prs.slides.add_slide(blank)
        text(s, title, 0.6, 0.35, 12.1, 0.9, size=30, bold=True, color=NAVY)
        foot = "Team 10 · ADSC × American Airlines · Fall 2026"
        if MODE == "FAKE":
            foot = "FAKE DATA — pipeline test, not results · " + foot
        text(s, f"{foot}   |   {count[0]}", 0.6, 7.0, 12.1, 0.4, size=10,
             color=RED if MODE == "FAKE" else GREY)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def bullets(s, items, left=0.7, top=1.4, width=11.9, height=5.4, size=20):
        return text(s, [f"•  {i}" for i in items], left, top, width, height, size=size)

    def image(s, name, left, top, width, height):
        path = FIGS / name
        if not path.exists():
            text(s, f"(figure missing: {name})", left, top, width, 0.5, size=14, color=RED)
            return
        w, h = Image.open(path).size
        scale = min(width / w, height / h)
        s.shapes.add_picture(str(path), Inches(left + (width - w * scale) / 2), Inches(top),
                             Inches(w * scale), Inches(h * scale))

    def table(s, header, rows, left, top, width, size=14):
        shape = s.shapes.add_table(len(rows) + 1, len(header), Inches(left), Inches(top),
                                   Inches(width), Inches(0.42 * (len(rows) + 1)))
        t = shape.table
        for j, h in enumerate(header):
            c = t.cell(0, j); c.text = h
            c.text_frame.paragraphs[0].font.size, c.text_frame.paragraphs[0].font.bold = Pt(size), True
        for i, r in enumerate(rows, start=1):
            for j, v in enumerate(r):
                c = t.cell(i, j); c.text = str(v)
                c.text_frame.paragraphs[0].font.size = Pt(size)

    carried, test, cov = d5["carried"], d5["test"], d5["coverage"]
    tc, tn = test[carried], test[NAIVE]
    o = d5["origins"]

    s = slide("Two-Year Passenger Demand Forecast")
    text(s, ["Team 10 — ADSC × American Airlines, Fall 2026", "Team members: (add names)",
             f"Built from the AA passenger and airport files · {TODAY}"], 0.7, 2.6, 11.5, 2.5, size=24)

    s = slide("The question", "Frame it as a planning tool: one forecast made today covering two years.")
    bullets(s, ["How many passengers will each origin–destination route carry in each of the next 24 months?",
                "The forecast is made once, today. It is a two-year plan, not a rolling next-month prediction.",
                "Use: network outlook and route-level capacity and schedule planning.",
                "Measure: route-month WAPE (total absolute error ÷ total passengers), reported separately for year 1 and year 2."])

    s = slide("Data and how we kept it safe")
    bullets(s, ["AA_PAX_FORECASTING: passengers per route per departure month. AA_AIRPORT_INFO: airport attributes.",
                "Dropped: same-month revenue (an outcome, not an input) and export columns (row number, load timestamp).",
                "The data never left team laptops: outputs are git-ignored, and only rounded percentages are reported.",
                "The pipeline was built and tested on generated fake data with the same columns before touching the real files."])

    s = slide("What demand looks like", "Point out the yearly cycle and the 2020 break; the shaded window is excluded from training pairs.")
    image(s, "eda_network_series.png", 0.6, 1.3, 12.1, 4.2)
    bullets(s, ["Strong yearly cycle; a 2020 collapse and recovery (shaded) is excluded from model training pairs."], top=5.7, height=1.2, size=18)

    s = slide("Seasonality differs by market")
    image(s, "eda_seasonality.png", 0.6, 1.3, 7.2, 5.4)
    bullets(s, ["Domestic and international routes peak differently.",
                "So the reference forecast repeats each route's own last year, month by month.",
                "Growth is modelled on top of that, never as a raw level."], left=8.0, width=4.8, size=18)

    s = slide("Approach")
    bullets(s, ["Reference: seasonal naive. Same route, same month, from the year before the forecast date, for both years.",
                "Challengers predict year-over-year growth on top of the naive, capped at 0.8×–1.25× per year:",
                "    trend-adjusted naive · Ridge regression · gradient-boosted trees",
                "Leakage controls: every input is known on the forecast date for all 24 months; no forecast is fed back as an input.",
                "Settings are fixed in code, and the rule for replacing the naive was written before any score existed."])

    s = slide("How we tested it")
    table(s, ["origin", "forecast from", "role"],
          [["stability", o["stability"], "check that a winner is not one lucky window"],
           ["select", o["select"], "main model comparison (D4)"],
           ["test", o["test"], "final two-year test (D5)"],
           ["forward", o["forward"], "the forecast for the next two years"]], 0.7, 1.5, 11.9, size=16)
    bullets(s, ["Each origin gets one 24-month forecast, never updated with actuals from inside the window.",
                "A challenger replaces the naive only if it wins year 1 AND year 2 at the select origin and is no worse at the stability origin."],
            top=4.0, height=2.8, size=18)

    s = slide("Model comparison (D4)")
    names = list(d4["scores"]["select"].keys())
    rows = []
    for n in names:
        a, b_ = d4["scores"]["select"].get(n), d4["scores"]["stability"].get(n)
        rows.append([n + ("  ← carried" if n == d4["carried"] else ""),
                     pct(a["WAPE_y1"]) if a else "n/a", pct(a["WAPE_y2"]) if a else "n/a",
                     pct(b_["WAPE_24m"]) if b_ else "n/a"])
    table(s, ["candidate", "select: WAPE y1", "select: WAPE y2", "stability: WAPE 24m"], rows, 0.7, 1.5, 11.9, size=16)
    image(s, "d4_wape_by_horizon.png", 1.5, 3.9, 10.3, 3.0)

    s = slide("Decision", "Explain the pre-registered rule; if the naive won, that is a legitimate, defensible result.")
    bullets(s, [f"Carried model: {d4['carried']}"] + d4["notes"], size=18)

    s = slide("Two-year test results (D5)")
    rows = [[f"{carried} (carried)", pct(tc["WAPE_y1"]), pct(tc["WAPE_y2"]), pct(tc["WAPE_24m"])]]
    if carried != NAIVE:
        rows.append([NAIVE, pct(tn["WAPE_y1"]), pct(tn["WAPE_y2"]), pct(tn["WAPE_24m"])])
    rows += [["bias (forecast ÷ actual − 1)", spct(tc["bias_y1"]), spct(tc["bias_y2"]), spct(tc["bias_24m"])],
             ["80% band coverage", pct(cov["y1"], 0), pct(cov["y2"], 0), pct(cov["all"], 0)]]
    table(s, ["", "year 1", "year 2", "both years"], rows, 0.7, 1.5, 11.9, size=16)
    verdict = blocks["d5_verdict"].split("**Verdict:**")[-1].strip()
    bullets(s, [f"Network-level monthly WAPE: {pct(tc['network_WAPE'])}", verdict], top=4.3, height=2.5, size=18)

    s = slide("Observed vs forecast on the test window")
    image(s, "d5_observed_vs_forecast.png", 0.6, 1.3, 12.1, 5.5)

    s = slide("Where the forecast is weaker")
    image(s, "d5_error_by_horizon.png", 0.4, 1.3, 6.2, 4.2)
    image(s, "d5_error_by_segment.png", 6.7, 1.3, 6.2, 4.2)
    lines = [l[2:].split(": ", 1)[-1] for l in blocks["figures"].splitlines()[1:3]]
    bullets(s, lines, top=5.6, height=1.3, size=15)

    s = slide("The next two years")
    image(s, "d5_forward_forecast.png", 0.6, 1.3, 12.1, 4.4)
    bullets(s, [f"Next 12 months vs last 12 observed: {spct(d5['forward']['y1_vs_last12'])}; "
                f"the following year vs that: {spct(d5['forward']['y2_vs_y1'])}. The shaded area is the 80% band."],
            top=5.9, height=1.0, size=18)

    s = slide("How American Airlines can use it")
    table(s, ["use", "supported?"],
          [["Network-level two-year demand outlook", "yes"],
           ["Route capacity / schedule planning, year 1", "large routes, with the segment errors in mind"],
           ["Route planning, year 2", "only as a baseline (error grows with horizon)"],
           ["New routes or closures", "no (forecasts only routes that exist today)"],
           ["Pricing / revenue", "no (no fare inputs by design)"]], 0.7, 1.5, 11.9, size=16)

    s = slide("Risks, monitoring and next steps")
    bullets(s, ["Risks: demand shocks break the growth cap; routes that open later are missed; only one 24-month test window exists.",
                "Monitor monthly: flag when the actual network total falls outside the band two months in a row.",
                "Re-run the comparison yearly; the naive needs no refitting.",
                "Data that would help most: scheduled seats per route-month, the planned route roster, lagged fares."], size=19)

    path = OUT / "team10_final_slides.pptx"
    prs.save(path)
    return path


def main() -> None:
    d4, d5 = load("d4_scores.json"), load("d5_scores.json")
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = build_blocks(d4, d5)
    for p in fill_reports(blocks):
        print("report:", p)
    deck = build_deck(d4, d5, blocks)
    if deck:
        print("slides:", deck)
    print(f"\nMODE = {MODE}. " + ("Review, then copy the reports into team-10-modeling/results/."
                                 if MODE == "REAL" else "FAKE build: do not submit."))


if __name__ == "__main__":
    main()
