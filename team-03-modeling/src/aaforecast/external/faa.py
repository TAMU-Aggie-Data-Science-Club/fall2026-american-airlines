"""FAA calendar-year enplanements by airport."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .http import download, get_text

log = logging.getLogger(__name__)
BASE = "https://www.faa.gov"
PAGES = [
    "/airports/planning_capacity/passenger_allcargo_stats/passenger",
    "/airports/planning_capacity/passenger_allcargo_stats/passenger/previous_years",
]
LINK_RE = re.compile(r'href="([^"]*all[-_]enplanements[^"]*)"', re.I)


def _discover_links(min_year: int) -> list[str]:
    """Find all-enplanements workbook links on the FAA pages, following year subpages."""
    found: set[str] = set()
    pages = list(PAGES)
    seen: set[str] = set()
    while pages:
        page = pages.pop()
        if page in seen:
            continue
        seen.add(page)
        try:
            html = get_text(BASE + page if page.startswith("/") else page)
        except Exception as exc:
            log.warning("FAA page %s unavailable: %s", page, exc)
            continue
        for link in LINK_RE.findall(html):
            m = re.search(r"cy(\d{2,4})", link, re.I)
            if not m:
                continue
            yr = int(m.group(1))
            yr = yr + 2000 if yr < 100 else yr
            if yr < min_year:
                continue
            if link.lower().endswith(".xlsx"):
                found.add(link if link.startswith("http") else BASE + link)
            elif not link.lower().endswith(".pdf"):
                pages.append(link)
    return sorted(found)


def _parse(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header_row = next(i for i in range(min(15, len(raw))) if raw.iloc[i].astype(str).str.fullmatch(r"(?i)locid").any())
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row]]
    year_cols = {c: int(re.search(r"(\d{2,4})", c).group(1)) for c in df.columns if re.search(r"(?i)cy\s*\d{2,4}\s*enplanements", c)}
    out = []
    for col, yr in year_cols.items():
        part = df[["Locid", col]].rename(columns={"Locid": "airport", col: "enplanements"})
        part["year"] = yr + 2000 if yr < 100 else yr
        out.append(part)
    res = pd.concat(out)
    res["enplanements"] = pd.to_numeric(res["enplanements"], errors="coerce")
    res["airport"] = res["airport"].astype(str).str.strip().str.upper()
    return res.dropna()


def enplanements(dest_dir: Path, force: bool = False, min_year: int = 2018) -> pd.DataFrame:
    """Long table: airport, year, enplanements (latest release wins on overlaps)."""
    frames = []
    for url in _discover_links(min_year):
        try:
            path = download(url, dest_dir / url.rsplit("/", 1)[-1], force=force)
            df = _parse(path)
            df["release"] = int(re.search(r"cy(\d{2,4})", url, re.I).group(1)[-2:])
            frames.append(df)
        except Exception as exc:
            log.warning("FAA file %s skipped: %s", url, exc)
    if not frames:
        raise RuntimeError("No FAA enplanement files could be read")
    df = pd.concat(frames).sort_values("release")
    return df.drop_duplicates(["airport", "year"], keep="last").drop(columns="release")
