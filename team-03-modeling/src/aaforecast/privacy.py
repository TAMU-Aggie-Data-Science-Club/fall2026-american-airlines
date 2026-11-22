"""Narrow source-tree artifact checks; human confidentiality review is still required."""
import json
from pathlib import Path
import re
import subprocess

BLOCKED_DIRS = {"data", "outputs", "runs", "artifacts", "models", "mlruns", "mlartifacts"}
BLOCKED_SUFFIXES = {".csv", ".gz", ".parquet", ".pickle", ".pkl", ".joblib", ".png", ".jpg", ".xlsx", ".pdf", ".zip", ".db", ".sqlite"}
TOKEN = re.compile(r"(?:sk-[A-Za-z0-9_-]{32,}|gh[pousr]_[A-Za-z0-9]{30,})")


def check_file(path, root):
    relative = Path(path).relative_to(root)
    reasons = []
    if {part.lower() for part in relative.parts} & BLOCKED_DIRS or relative.suffix.lower() in BLOCKED_SUFFIXES:
        reasons.append("restricted artifact path/type")
    name = relative.name.lower()
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        reasons.append("environment secrets file")
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (UnicodeError, OSError):
        return reasons + ["unreviewed binary/unreadable file"]
    if TOKEN.search(text):
        reasons.append("credential-like content")
    if relative.suffix.lower() == ".ipynb":
        try:
            notebook = json.loads(text)
            if any(cell.get("outputs") for cell in notebook.get("cells", [])):
                reasons.append("notebook contains outputs")
        except ValueError:
            reasons.append("invalid notebook")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4:
        rows = [line.split(",") for line in lines[:10]]
        if all(len(row) >= 4 for row in rows) and len({len(row) for row in rows}) == 1:
            try:
                for row in rows[1:]:
                    for value in row:
                        float(value)
                reasons.append("dataset-like content under another extension")
            except ValueError:
                pass
    return reasons


def scan_repository(root):
    root = Path(root)
    result = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
    failures = {}
    for name in result.decode().split("\0"):
        path = root / name
        if name and path.is_file():
            reasons = check_file(path, root)
            if reasons:
                failures[name] = reasons
    return failures


if __name__ == "__main__":
    import sys
    failures = scan_repository(Path.cwd())
    for path, reasons in failures.items():
        print(path + ": " + "; ".join(reasons))
    print("Source-tree guard: " + ("FAIL" if failures else "PASS") + ". Human review required; this cannot detect every confidential artifact.")
    sys.exit(bool(failures))
