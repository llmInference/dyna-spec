from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sglang as sgl
from .vocab_manager import vocab_manager

# Assume the SGLang runtime is globally initialized and exposes a tokenizer.
tokenizer = getattr(getattr(sgl, "global_runtime", None), "tokenizer", None)

app = FastAPI()


class VocabRequest(BaseModel):
    client_id: str
    words: List[str]


class GenerateRequest(BaseModel):
    client_id: str
    prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None


def _ensure_tokenizer():
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not initialized")
    return tokenizer


@app.post("/v1/vocab/add")
def add_vocab(req: VocabRequest):
    tok = _ensure_tokenizer()
    token_ids = tok.convert_tokens_to_ids(req.words)
    vocab_manager.add_words(req.client_id, token_ids)
    return {"status": "success"}


@app.post("/v1/vocab/remove")
def remove_vocab(req: VocabRequest):
    tok = _ensure_tokenizer()
    token_ids = tok.convert_tokens_to_ids(req.words)
    vocab_manager.remove_words(req.client_id, token_ids)
    return {"status": "success"}


@app.post("/v1/generate")
def generate(req: GenerateRequest):
    active_vocab_ids = vocab_manager.get_vocab_list(req.client_id)
    if not active_vocab_ids:
        raise HTTPException(
            status_code=400, detail=f"No active vocabulary for client {req.client_id}"
        )

    generate_kwargs = {
        "prompt": req.prompt,
        "dynamic_vocab_token_ids": active_vocab_ids,
    }
    if req.max_new_tokens is not None:
        generate_kwargs["max_new_tokens"] = req.max_new_tokens
    if req.temperature is not None:
        generate_kwargs["temperature"] = req.temperature

    state = sgl.gen(**generate_kwargs)
    return {"text": state.text()}

