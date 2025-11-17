#!/usr/bin/env python3
"""Create a sampled token-frequency report from an existing vocab file."""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Optional

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a fraction of tokens from an existing token_freq file (e.g., SlimPajama 10%).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Source token_freq file (token_id\\tcount[\\ttoken_text]).",
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the sampled frequency file."
    )
    parser.add_argument(
        "--ratio", type=float, default=0.1, help="Fraction of tokens to keep (0-1]."
    )
    parser.add_argument(
        "--strategy",
        choices=["top", "random"],
        default="top",
        help="Sampling strategy: top keeps the most frequent tokens, random selects uniformly.",
    )
    parser.add_argument(
        "--seed", type=int, default=13, help="Random seed when --strategy=random."
    )
    parser.add_argument(
        "--freq-include-token-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include decoded token text as the third column when possible.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name/path for decoding token ids (required if input lacks token text and --freq-include-token-text is true).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer from local cache only (if --tokenizer specified).",
    )
    parser.add_argument(
        "--vocab-output",
        default=None,
        help="Optional path to write the sampled static vocab (ids only, sorted ascending).",
    )
    return parser.parse_args()


def _format_token_text(tokenizer, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:  # pragma: no cover - tokenizer-specific edge cases
        token = None
    if token is None:
        return "<unk>"
    return token.encode("unicode_escape").decode("utf-8")


def _read_entries(path: str) -> List[dict]:
    entries: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                token_id = int(parts[0])
                count = int(parts[1])
            except ValueError:
                continue
            token_text: Optional[str] = parts[2] if len(parts) > 2 else None
            entries.append(
                {"token_id": token_id, "count": count, "token_text": token_text}
            )
    return entries


def _write_frequency(
    entries: List[dict], path: str, tokenizer, include_token_text: bool
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = sorted(
        entries, key=lambda item: (-item["count"], item["token_id"])
    )
    with open(path, "w", encoding="utf-8") as f:
        for item in sorted_entries:
            token_id = item["token_id"]
            count = item["count"]
            token_text = item.get("token_text")
            if include_token_text:
                if token_text is None and tokenizer is not None:
                    token_text = _format_token_text(tokenizer, token_id)
                elif token_text is None:
                    token_text = ""
                f.write(f"{token_id}\t{count}\t{token_text}\n")
            else:
                f.write(f"{token_id}\t{count}\n")


def _write_vocab(entries: List[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    token_ids = sorted({item["token_id"] for item in entries})
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for token_id in token_ids:
            f.write(f"{token_id}\n")
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )
    if not (0 < args.ratio <= 1.0):
        raise ValueError("--ratio must be between 0 and 1")
    entries = _read_entries(args.input)
    if not entries:
        raise SystemExit(f"No entries parsed from {args.input}")
    target_count = max(1, math.floor(len(entries) * args.ratio))
    sorted_entries = sorted(
        entries, key=lambda item: (-item["count"], item["token_id"])
    )
    if args.strategy == "top":
        sampled = sorted_entries[:target_count]
    else:
        rng = random.Random(args.seed)
        sampled = rng.sample(sorted_entries, target_count)
    tokenizer = None
    if args.freq_include_token_text and any(
        item.get("token_text") is None for item in sampled
    ):
        if args.tokenizer is None:
            raise ValueError(
                "--tokenizer is required to decode token ids when freq output should include token text."
            )
        if AutoTokenizer is None:
            raise SystemExit("transformers is required to decode token ids.")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            use_fast=True,
            trust_remote_code=False,
            local_files_only=args.local_files_only,
        )
    logger.info(
        "Sampling %d / %d tokens (ratio=%.4f, strategy=%s) from %s",
        len(sampled),
        len(entries),
        args.ratio,
        args.strategy,
        args.input,
    )
    _write_frequency(sampled, args.output, tokenizer, args.freq_include_token_text)
    logger.info("Sampled frequency file written to %s", args.output)
    if args.vocab_output:
        _write_vocab(sampled, args.vocab_output)
        logger.info("Sampled vocab file written to %s", args.vocab_output)


if __name__ == "__main__":
    main()
