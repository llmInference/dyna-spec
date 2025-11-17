#!/usr/bin/env python3
"""Merge multiple token frequency files into a single report (and optional vocab)."""
from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore

logger = logging.getLogger(__name__)


def _format_token_text(tokenizer, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:  # pragma: no cover - tokenizer-specific edge cases
        token = None
    if token is None:
        return "<unk>"
    return token.encode("unicode_escape").decode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge token frequency files produced by stream_slimpajama_vocab.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Frequency files to merge (token_id TAB count per line).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the merged frequency file.",
    )
    parser.add_argument(
        "--annotate-token-text",
        action="store_true",
        help="When set, include a third column with decoded token text (requires --tokenizer).",
    )
    parser.add_argument(
        "--vocab-output",
        default=None,
        help="Optional path to write a static vocab (top-k ids sorted ascending).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Top-k tokens to keep when emitting vocab (requires --vocab-output).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name/path for vocab size validation or token annotation (required if --vocab-output or --annotate-token-text is set).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer from local cache only (if --tokenizer specified).",
    )
    return parser.parse_args()


def load_freq_file(path: str) -> Counter:
    counter: Counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split("\t")
                token_str, count_str = parts[0], parts[1]
                token_id = int(token_str)
                count = int(count_str)
            except (ValueError, IndexError):
                logger.warning("Skipping malformed line in %s: %s", path, line)
                continue
            counter[token_id] += count
    return counter


def write_frequency_file(
    counter: Counter, path: str, tokenizer=None, include_token_text: bool = False
) -> None:
    freq_lines = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for token_id, count in freq_lines:
            if include_token_text and tokenizer is not None:
                token_text = _format_token_text(tokenizer, token_id)
                f.write(f"{token_id}\t{count}\t{token_text}\n")
            else:
                f.write(f"{token_id}\t{count}\n")


def write_vocab(
    counter: Counter, path: str, topk: int, tokenizer_name: str, local_only: bool
) -> None:
    if AutoTokenizer is None:
        raise SystemExit("transformers is required to emit vocab output.")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
        local_files_only=local_only,
    )
    vocab_size = tokenizer.vocab_size
    most_common = counter.most_common(topk)
    filtered = [tid for tid, _ in most_common if 0 <= tid < vocab_size]
    filtered_sorted = sorted(filtered)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for tid in filtered_sorted:
            f.write(f"{tid}\n")
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    if args.annotate_token_text and args.tokenizer is None:
        raise ValueError("--tokenizer is required when --annotate-token-text is set.")
    merged_counter: Counter = Counter()
    for path in tqdm(args.inputs, desc="Merging freq files"):
        merged_counter.update(load_freq_file(path))
    logger.info("Writing merged frequency file to %s", args.output)
    annotation_tokenizer = None
    if args.annotate_token_text:
        if AutoTokenizer is None:
            raise SystemExit(
                "transformers is required when --annotate-token-text is set."
            )
        annotation_tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            use_fast=True,
            trust_remote_code=False,
            local_files_only=args.local_files_only,
        )
    write_frequency_file(
        merged_counter,
        args.output,
        tokenizer=annotation_tokenizer,
        include_token_text=args.annotate_token_text,
    )

    if args.vocab_output:
        if args.topk is None or args.tokenizer is None:
            raise ValueError(
                "--topk and --tokenizer are required when --vocab-output is set."
            )
        logger.info(
            "Writing merged static vocab to %s (topk=%d)",
            args.vocab_output,
            args.topk,
        )
        write_vocab(
            merged_counter,
            args.vocab_output,
            args.topk,
            args.tokenizer,
            args.local_files_only,
        )


if __name__ == "__main__":
    main()
