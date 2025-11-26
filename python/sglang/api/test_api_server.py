import random
from typing import Any, Dict

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


class FakeResult:
    def __init__(self, text: str, meta_info: Dict[str, Any] | None = None):
        self._text = text
        self._meta = meta_info or {"avg_spec_accept_length": 0.0}

    def text(self):
        return self._text

    def meta_info(self):
        return self._meta


class FakeLogitsProbe:
    """Simulate backend logits processor behavior."""

    def __init__(self, vocab_size: int = 32768, seed: int = 1234):
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)
        self.last_projected_ids = None
        self.last_logits_vector = None

    def project(self, token_ids):
        if token_ids is None:
            raise AssertionError("dynamic vocab ids must be provided")
        if len(token_ids) == 0:
            raise AssertionError("dynamic vocab ids must not be empty")
        assert all(
            0 <= tid < self.vocab_size for tid in token_ids
        ), "token ids must be within base vocab"

        # Simulate a full-vocab logits vector and then select the dynamic subset.
        full_logits = [self.rng.random() for _ in range(self.vocab_size)]
        subset_logits = [full_logits[tid] for tid in token_ids]

        self.last_projected_ids = list(token_ids)
        self.last_logits_vector = subset_logits
        return subset_logits


class FakeRuntime:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.calls = []
        self.logits_probe = FakeLogitsProbe()
        self.vocab_size = 32768
        self.last_hidden_states = None

    def generate(
        self,
        prompt,
        *,
        sampling_params=None,
        return_logprob=False,
        logprob_start_len=None,
        top_logprobs_num=None,
        lora_path=None,
        return_hidden_states=False,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "return_logprob": return_logprob,
                "logprob_start_len": logprob_start_len,
                "top_logprobs_num": top_logprobs_num,
                "lora_path": lora_path,
                "return_hidden_states": return_hidden_states,
            }
        )
        dynamic_ids = sampling_params.get("dynamic_vocab_token_ids")
        self.logits_probe.project(dynamic_ids)
        meta = {"avg_spec_accept_length": 0.0}
        if return_hidden_states:
            self.last_hidden_states = [[float(i), float(i + 1)] for i in range(3)]
            meta["hidden_states"] = self.last_hidden_states
        else:
            self.last_hidden_states = None
        return FakeResult("generated text", meta_info=meta)

    def get_tokenizer(self):
        return self.tokenizer
    
    def _get_model_info(self, force_refresh=False):
        return {
            "vocab_size": self.vocab_size,
            "use_static_vocab": False,
        }
    
    def get_server_info(self):
        return {
            "vocab_size": self.vocab_size,
            "init_vocab_size": 327,  # Simulate initial vocab size
        }


@pytest.fixture
def api_client(monkeypatch):
    from sglang.api import api_server

    backend = FakeRuntime()
    prev_backend = sglang.global_config.default_backend
    sglang.set_default_backend(backend)

    manager = ActiveVocabManager()
    monkeypatch.setattr(api_server, "vocab_manager", manager, raising=False)
    monkeypatch.setattr(api_pkg, "vocab_manager", manager, raising=False)
    monkeypatch.setattr(vocab_module, "vocab_manager", manager, raising=False)

    client = TestClient(api_server.app)
    yield client, manager, backend.tokenizer, backend
    sglang.set_default_backend(prev_backend)


def test_add_and_generate_flow(api_client):
    client, manager, tokenizer, backend = api_client

    resp = client.post(
        "/v1/vocab/add", json={"words": ["alpha", "beta"]}
    )
    assert resp.status_code == 200
    assert manager.get_vocab_list()

    resp = client.post(
        "/v1/generate",
        json={
            "prompt": "Hello",
            "max_new_tokens": 5,
            "temperature": 0.5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "generated text"
    assert data["draft_avg_accept_length"] == 0.0
    assert "meta_info" in data
    call = backend.calls[-1]
    assert call["prompt"] == "Hello"
    params = call["sampling_params"]
    assert params["max_new_tokens"] == 5
    assert params["temperature"] == 0.5
    assert params["dynamic_vocab_token_ids"] == manager.get_vocab_list()


def test_remove_vocab_updates_manager(api_client):
    client, manager, tokenizer, _ = api_client

    client.post(
        "/v1/vocab/add", json={"words": ["gamma", "delta"]}
    )
    gamma_id = tokenizer.convert_tokens_to_ids(["gamma"])[0]

    resp = client.post(
        "/v1/vocab/remove", json={"words": ["gamma"]}
    )
    assert resp.status_code == 200
    remaining = manager.get_vocab_list()
    assert gamma_id not in remaining


def test_generate_without_vocab_uses_initial_vocab(api_client):
    client, manager, _, backend = api_client

    # Query once to trigger initialization
    status = client.get("/v1/vocab/query").json()
    assert status["vocab_count"] == status["initial_vocab_count"]
    assert status["vocab_count"] == manager.get_initial_vocab_size()
    assert status["vocab_count"] > 0

    resp = client.post("/v1/generate", json={"prompt": "No vocab"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "generated text"

    expected_ids = manager.get_vocab_list()
    assert expected_ids
    last_call = backend.calls[-1]
    assert last_call["sampling_params"]["dynamic_vocab_token_ids"] == expected_ids


def test_dynamic_vocab_reaches_logits_probe(api_client):
    client, manager, _, backend = api_client

    new_words = ["omega", "sigma", "lambda"]
    resp = client.post(
        "/v1/vocab/add", json={"words": new_words}
    )
    assert resp.status_code == 200

    resp = client.post(
        "/v1/generate",
        json={
            "prompt": "Testing dynamic vocab path",
            "max_new_tokens": 4,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "generated text"
    assert data["draft_avg_accept_length"] == 0.0
    assert "meta_info" in data

    expected_ids = manager.get_vocab_list()
    assert expected_ids  # Should not be empty
    assert backend.logits_probe.last_projected_ids == expected_ids
    assert len(backend.logits_probe.last_logits_vector) == len(expected_ids)


def test_hidden_states_endpoint_returns_meta(api_client):
    client, manager, _, backend = api_client

    client.post("/v1/vocab/add", json={"words": ["theta"]})
    resp = client.post(
        "/v1/hidden_states",
        json={
            "prompt": "Need hidden states",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["hidden_states"] == backend.last_hidden_states
    assert data["meta_info"]["hidden_states"] == backend.last_hidden_states
    assert data["meta_info"]["avg_spec_accept_length"] == 0.0

