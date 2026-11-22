"""Small cached downloader for public data sources."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "aaforecast-research/0.1 (python-requests)"}
ALLOWED_HOSTS = frozenset({
    "www.faa.gov", "faa.gov", "www2.census.gov", "www.census.gov",
    "fred.stlouisfed.org", "public.opendatasoft.com",
    "davidmegginson.github.io", "raw.githubusercontent.com",
})


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443)):
        raise ValueError("Public download URL must use HTTPS on an approved source host")


def _response(url: str, timeout: int):
    """Validate every redirect before opening it; callers close the response."""
    for _ in range(6):
        _validate_url(url)
        response = requests.get(url, headers=HEADERS, timeout=timeout,
                                stream=True, allow_redirects=False)
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ValueError("Public source returned a redirect without a location")
            url = urljoin(url, location)
            continue
        try:
            response.raise_for_status()
        except Exception:
            response.close()
            raise
        return response
    raise ValueError("Public source exceeded the redirect limit")


def _chunks(response, max_bytes: int):
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    declared = response.headers.get("Content-Length")
    if declared and int(declared) > max_bytes:
        raise ValueError("Public source exceeds the download size limit")
    total = 0
    for chunk in response.iter_content(chunk_size=1 << 20):
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("Public source exceeds the download size limit")
        yield chunk


def download(url: str, dest: Path, force: bool = False, retries: int = 3,
             timeout: int = 120, max_bytes: int = 200 * 1024 * 1024) -> Path:
    """Download ``url`` to ``dest`` unless it already exists."""
    _validate_url(url)
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with _response(url, timeout) as r:
                with open(tmp, "wb") as fh:
                    for chunk in _chunks(r, max_bytes):
                        fh.write(chunk)
            tmp.replace(dest)
            log.info("Downloaded %s", url)
            return dest
        except requests.RequestException as exc:
            log.warning("Attempt %d for %s failed: %s", attempt, url, exc)
            if attempt < retries:
                time.sleep(2 * attempt)
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {url}")


def get_text(url: str, timeout: int = 60, max_bytes: int = 10 * 1024 * 1024) -> str:
    with _response(url, timeout) as r:
        body = b"".join(_chunks(r, max_bytes))
        return body.decode(r.encoding or "utf-8", errors="replace")
