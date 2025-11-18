from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import sglang
import sglang.api as api_pkg
import sglang.api.vocab_manager as vocab_module
from sglang.api.vocab_manager import ActiveVocabManager


class DummyTokenizer:
    def __init__(self):
        self._token_map = {}
        self._next_id = 1

    def convert_tokens_to_ids(self, words):
        ids = []
        for word in words:
            if word not in self._token_map:
                self._token_map[word] = self._next_id
                self._next_id += 1
            ids.append(self._token_map[word])
        return ids


@pytest.fixture
def api_client(monkeypatch):
    from sglang.api import api_server

    tokenizer = DummyTokenizer()
    monkeypatch.setattr(api_server, "tokenizer", tokenizer, raising=False)

    calls = {}

    def fake_generate(**kwargs):
        calls["kwargs"] = kwargs
        return SimpleNamespace(text=lambda: "generated text")

    monkeypatch.setattr(api_server.sgl, "gen", fake_generate)

    manager = ActiveVocabManager()
    monkeypatch.setattr(api_server, "vocab_manager", manager, raising=False)
    monkeypatch.setattr(api_pkg, "vocab_manager", manager, raising=False)
    monkeypatch.setattr(vocab_module, "vocab_manager", manager, raising=False)

    client = TestClient(api_server.app)
    return client, manager, tokenizer, calls


def test_add_and_generate_flow(api_client):
    client, manager, tokenizer, calls = api_client

    resp = client.post(
        "/v1/vocab/add", json={"client_id": "client-1", "words": ["alpha", "beta"]}
    )
    assert resp.status_code == 200
    assert manager.get_vocab_list("client-1")

    resp = client.post(
        "/v1/generate",
        json={
            "client_id": "client-1",
            "prompt": "Hello",
            "max_new_tokens": 5,
            "temperature": 0.5,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "generated text"}
    assert calls["kwargs"]["prompt"] == "Hello"
    assert calls["kwargs"]["max_new_tokens"] == 5
    assert calls["kwargs"]["temperature"] == 0.5
    assert calls["kwargs"]["dynamic_vocab_token_ids"] == manager.get_vocab_list(
        "client-1"
    )


def test_remove_vocab_updates_manager(api_client):
    client, manager, tokenizer, _ = api_client

    client.post(
        "/v1/vocab/add", json={"client_id": "client-2", "words": ["gamma", "delta"]}
    )
    gamma_id = tokenizer.convert_tokens_to_ids(["gamma"])[0]

    resp = client.post(
        "/v1/vocab/remove", json={"client_id": "client-2", "words": ["gamma"]}
    )
    assert resp.status_code == 200
    remaining = manager.get_vocab_list("client-2")
    assert gamma_id not in remaining


def test_generate_without_vocab_fails(api_client):
    client, *_ = api_client

    resp = client.post(
        "/v1/generate", json={"client_id": "missing", "prompt": "No vocab"}
    )
    assert resp.status_code == 400
    assert "No active vocabulary" in resp.json()["detail"]

