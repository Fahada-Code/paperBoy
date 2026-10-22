"""Tests for OllamaBackend.embed_many batching (httpx.post is faked, no network)."""

import httpx

from paperboy.backends.ollama_backend import OllamaBackend


class _Resp:
    def __init__(self, status: int, data: dict):
        self.status_code, self._data, self.text = status, data, ""

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


def test_embed_many_batches_and_prefixes(monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json["input"])
        return _Resp(200, {"embeddings": [[float(len(t))] for t in json["input"]]})

    monkeypatch.setattr(httpx, "post", fake_post)
    b = OllamaBackend("m")
    b.EMBED_BATCH = 2
    out = b.embed_many(["a", "bb", "ccc"], prefix="p:")
    assert sent == [["p:a", "p:bb"], ["p:ccc"]]
    assert out == [[3.0], [4.0], [5.0]]


def test_embed_many_falls_back_when_batch_endpoint_missing(monkeypatch):
    def fake_post(url, json, timeout):
        if url.endswith("/api/embed"):
            return _Resp(404, {})
        return _Resp(200, {"embedding": [1.0]})  # legacy /api/embeddings

    monkeypatch.setattr(httpx, "post", fake_post)
    assert OllamaBackend("m").embed_many(["x", "y"]) == [[1.0], [1.0]]
