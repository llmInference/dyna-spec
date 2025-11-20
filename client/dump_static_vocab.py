#!/usr/bin/env python3
"""Utility script to dump the first N tokens from the active tokenizer into JSON.

This script mirrors the tokenizer loading logic used by the API server so it can
run either when a local runtime backend is configured (via SGLANG_RUNTIME_URL) or
when ``sglang.set_default_backend(...)`` has already been called in the current
Python environment.

Example:
    python client/dump_static_vocab.py --count 1600 --output vocab_top1600.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import sglang as sgl


def _ensure_tokenizer(runtime_url: Optional[str]) -> Any:
    """Load the tokenizer using the same fallbacks as the API server."""

    if runtime_url:
        os.environ.setdefault("SGLANG_RUNTIME_URL", runtime_url)

    tokenizer = None
    last_error: Optional[Exception] = None

    try:
        tokenizer = sgl.get_tokenizer()
    except Exception as exc:
        last_error = exc

    if tokenizer is None:
        try:
            # Reuse the HTTP runtime fallback from the API server.
            from sglang.api.api_server import _get_http_backend  # type: ignore

            backend = _get_http_backend()
            if backend is not None:
                tokenizer = backend.get_tokenizer()
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    if tokenizer is None:
        detail = (
            f"Last error: {last_error}"
            if last_error is not None
            else "No tokenizer available from the current backend."
        )
        raise RuntimeError(
            "Unable to initialize tokenizer. Make sure a runtime backend is "
            "running and either call sglang.set_default_backend(...) or set "
            "SGLANG_RUNTIME_URL to point at the runtime server. "
            + detail
        )

    return tokenizer


def _dump_tokens(tokenizer: Any, count: int) -> List[Dict[str, Any]]:
    """Return the first ``count`` tokens as [{id, token}]."""

    if not hasattr(tokenizer, "convert_ids_to_tokens"):
        raise AttributeError("Tokenizer does not expose convert_ids_to_tokens")

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        if hasattr(tokenizer, "vocab"):
            vocab_size = len(tokenizer.vocab)
        elif hasattr(tokenizer, "get_vocab"):
            vocab_size = len(tokenizer.get_vocab())

    if vocab_size is None:
        raise AttributeError("Tokenizer does not provide vocab_size information")

    limit = min(count, int(vocab_size))
    tokens: List[Dict[str, Any]] = []

    for token_id in range(limit):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        tokens.append({"token_id": token_id, "token": token_str})

    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump the first N tokenizer entries into JSON."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1600,
        help="Number of tokens to dump (default: 1600).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("client/vocab_top1600.json"),
        help="Path to the JSON file to write.",
    )
    parser.add_argument(
        "--runtime-url",
        default=os.environ.get("SGLANG_RUNTIME_URL"),
        help="Optional runtime URL if one is not already configured.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = _ensure_tokenizer(args.runtime_url)
    tokens = _dump_tokens(tokenizer, args.count)

    payload = {
        "count": len(tokens),
        "runtime_url": args.runtime_url,
        "tokens": tokens,
    }

    output_path = args.output if isinstance(args.output, Path) else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=args.indent)

    print(f"Wrote {len(tokens)} tokens to {output_path}")


if __name__ == "__main__":
    main()

