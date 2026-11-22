"""Offline regression checks for public-source request boundaries."""
import pytest

from aaforecast.external import http


class Response:
    def __init__(self, status=200, headers=None, chunks=(b"ok",)):
        self.status_code = status
        self.headers = headers or {}
        self.encoding = "utf-8"
        self.chunks = chunks
        self.closed = False

    def close(self):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        return iter(self.chunks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.mark.parametrize("url", ["http://www.faa.gov/a", "https://127.0.0.1/a",
    "https://www.faa.gov.evil.example/a", "https://user:pass@www.faa.gov/a",
    "https://www.faa.gov:8443/a"])
def test_unapproved_destination_is_rejected_before_request(monkeypatch, url):
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: pytest.fail("request opened"))
    with pytest.raises(ValueError):
        http.get_text(url)


def test_redirect_is_checked_before_next_request(monkeypatch):
    response = Response(302, {"Location": "http://127.0.0.1/private"})
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        assert kwargs["allow_redirects"] is False
        return response
    monkeypatch.setattr(http.requests, "get", get)
    with pytest.raises(ValueError):
        http.get_text("https://www.faa.gov/start")
    assert calls == ["https://www.faa.gov/start"]
    assert response.closed


def test_size_limit_cleans_partial_file_and_preserves_destination(monkeypatch, tmp_path):
    response = Response(chunks=(b"abc", b"def"))
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: response)
    destination = tmp_path / "source.csv"
    destination.write_bytes(b"previous")
    with pytest.raises(ValueError):
        http.download("https://www.faa.gov/source.csv", destination, force=True, max_bytes=4)
    assert destination.read_bytes() == b"previous"
    assert not destination.with_suffix(".csv.part").exists()
    assert response.closed


def test_relative_redirect_and_text_success(monkeypatch):
    responses = iter([Response(302, {"Location": "/final"}), Response(chunks=(b"hello",))])
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: next(responses))
    assert http.get_text("https://www.faa.gov/start") == "hello"


def test_declared_oversize_response_rejected_without_reading(monkeypatch):
    response = Response(headers={"Content-Length": "100"})
    response.iter_content = lambda **kwargs: pytest.fail("oversize body was read")
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: response)
    with pytest.raises(ValueError):
        http.get_text("https://www.faa.gov/source", max_bytes=10)
    assert response.closed
