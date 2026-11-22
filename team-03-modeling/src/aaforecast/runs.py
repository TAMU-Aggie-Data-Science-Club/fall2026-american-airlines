"""Local immutable run identities and compatibility checks (no hosted tracking)."""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

from . import config


def input_identity():
    identities = {}
    for name, path in (("passengers", config.PAX_FILE), ("airports", config.AIRPORT_FILE)):
        if not path.exists():
            identities[name] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        identities[name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return identities


def source_fingerprint(directory):
    """Match Python's universal-newline handling across Windows/Linux checkouts."""
    code = hashlib.sha256()
    directory = Path(directory)
    for path in sorted(directory.rglob("*.py")):
        code.update(path.relative_to(directory).as_posix().encode("utf-8"))
        code.update(b"\0")
        code.update(path.read_text(encoding="utf-8-sig").encode("utf-8"))
        code.update(b"\0")
    return code.hexdigest()


def signature(cfg):
    settings = asdict(cfg)
    settings.pop("minimal_forecast", None)
    external = {}
    if cfg.use_external:
        for path in sorted((config.EXTERNAL_DIR / "processed").glob("*.csv")):
            external[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"config": settings, "inputs": input_identity(), "externals": external,
            "source_sha256": source_fingerprint(Path(__file__).parent)}


@contextmanager
def run_directory(kind, cfg):
    directory = config.OUTPUT_DIR / kind / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=config.ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    manifest = {"kind": kind, "status": "running", "revision": revision, "working_tree_dirty": dirty,
                "python": platform.python_version(), "started": datetime.now(timezone.utc).isoformat(), **signature(cfg)}
    def save():
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    try:
        yield directory
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    else:
        manifest["status"] = "complete"
    finally:
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        save()


def validate_backtest(directory, cfg):
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "backtest" or manifest.get("status") != "complete":
        raise ValueError("Calibration requires a completed backtest run")
    current = signature(cfg)
    if any(manifest.get(key) != json.loads(json.dumps(value)) for key, value in current.items()):
        raise ValueError("Backtest configuration or input fingerprint differs from this forecast")
