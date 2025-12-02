#!/usr/bin/env python3
"""Stream OpenWebText, tokenize with a draft tokenizer, and export frequency + static vocab files.

The script follows the requirements laid out in README_Dyna.md "附加需求：基于大语料按出现率生成静态词汇表" but now defaults to OpenWebText for faster experimentation.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import datasets
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - transformers is an optional dep
    raise SystemExit(
        "transformers is required for this script. Install via `pip install transformers`."
    ) from exc

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream SlimPajama, count token frequencies, and emit static vocab files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="Skylion007/openwebtext",
        help="HuggingFace dataset repo id (supports streaming).",
    )
    parser.add_argument(
        "--trust-dataset-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to datasets.load_dataset (needed for community datasets like OpenWebText).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to iterate over (streaming mode).",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or local path matching the draft model vocab.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Column/key containing raw text. For nested dicts use dot notation (e.g. meta.text).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of records to tokenize per batch (controls tokenizer throughput).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples for quick sanity checks.",
    )
    parser.add_argument(
        "--freq-output",
        required=True,
        help="Path to write token frequency file (token_id\tcount sorted by count desc).",
    )
    parser.add_argument(
        "--vocab-output",
        required=True,
        help="Path to write final static vocab (one token id per line, sorted ascending).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=True,
        help="Number of most frequent token ids to keep in the static vocab output.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Pickle file for persisting the Counter so runs can resume after interruption.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Save checkpoint every N processed samples (0 to disable).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Emit progress logs every N batches.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HuggingFace Hub token for private datasets.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code to the tokenizer loader.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--resume-offset",
        type=int,
        default=0,
        help="Skip the first N samples (useful when resuming manually).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer weights without hitting the network (if already cached).",
    )
    parser.add_argument(
        "--shard-total",
        type=int,
        default=1,
        help="Split the dataset into N shards for parallel runs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Process only this shard index (0-based).",
    )
    parser.add_argument(
        "--use-gpu-bincount",
        action="store_true",
        help="Use torch.bincount on CUDA to accumulate token frequencies.",
    )
    return parser.parse_args()


def _get_nested_text(record: dict, key: str) -> Optional[str]:
    parts = key.split(".")
    value = record
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    if isinstance(value, str):
        return value
    return None


def load_counter(path: Optional[str]) -> Counter:
    if path and os.path.exists(path):
        logger.info("Loading checkpointed counter from %s", path)
        with open(path, "rb") as f:
            return pickle.load(f)
    return Counter()


def save_counter(counter: Counter, path: Optional[str]) -> None:
    if not path:
        return
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(counter, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    logger.info("Checkpoint saved to %s", path)


def stream_dataset(
    dataset: str,
    split: str,
    revision: Optional[str],
    hf_token: Optional[str],
    trust_remote_code: bool,
) -> datasets.iterable_dataset.IterableDataset:
    ds = datasets.load_dataset(
        dataset,
        split=split,
        streaming=True,
        revision=revision,
        use_auth_token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    return ds


def tokenize_batch(tokenizer, texts: List[str]) -> List[int]:
    if not texts:
        return []
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids: List[int] = []
    # tokenizer output can be dict or BatchEncoding
    token_lists = encoded["input_ids"]
    for seq in token_lists:
        ids.extend(seq)
    return ids


def write_frequency_file(counter: Counter, path: str) -> None:
    logger.info("Writing frequency file to %s", path)
    freq_lines = sorted(
        counter.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for token_id, count in freq_lines:
            f.write(f"{token_id}\t{count}\n")


def write_vocab_file(counter: Counter, path: str, topk: int, vocab_size: int) -> None:
    logger.info(
        "Writing static vocab file to %s (topk=%d, vocab_size=%d)",
        path,
        topk,
        vocab_size,
    )
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
    if args.shard_total <= 0:
        raise ValueError("--shard-total must be >= 1")
    if not (0 <= args.shard_index < args.shard_total):
        raise ValueError("--shard-index must satisfy 0 <= index < shard_total")
    logger.info("Loading tokenizer %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    torch_device = None
    torch_module = None
    if args.use_gpu_bincount:
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "PyTorch is required when --use-gpu-bincount is set."
            ) from exc
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA device not available but --use-gpu-bincount was requested."
            )
        torch_device = torch.device("cuda")
        torch_module = torch
        logger.info("Using GPU bincount on device %s", torch_device)

    counter = load_counter(args.checkpoint)
    dataset_stream = stream_dataset(
        args.dataset,
        args.split,
        args.revision,
        args.hf_token,
        args.trust_dataset_code,
    )
    if args.shard_total > 1:
        dataset_stream = dataset_stream.shard(
            num_shards=args.shard_total, index=args.shard_index
        )
        logger.info("Processing shard %d / %d", args.shard_index, args.shard_total)

    batch: List[str] = []
    total_processed = 0
    skipped = 0
    progress = tqdm(disable=False, unit="samples", dynamic_ncols=True)

    for record in dataset_stream:
        if args.max_samples is not None and total_processed >= args.max_samples:
            break
        if skipped < args.resume_offset:
            skipped += 1
            continue
        text = _get_nested_text(record, args.text_key)
        if not text:
            continue
        batch.append(text)
        total_processed += 1
        progress.update(1)
        if len(batch) >= args.batch_size:
            token_ids = tokenize_batch(tokenizer, batch)
            if token_ids:
                if args.use_gpu_bincount:
                    assert torch_module is not None
                    token_tensor = torch_module.tensor(
                        token_ids, device=torch_device, dtype=torch_module.int64
                    )
                    bincount = torch_module.bincount(
                        token_tensor,
                        minlength=tokenizer.vocab_size,
                    )
                    nonzero = (
                        (bincount > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
                    )
                    for tid in nonzero:
                        counter[tid] += int(bincount[tid].item())
                else:
                    counter.update(token_ids)
            batch.clear()
        if (
            args.checkpoint_interval > 0
            and total_processed % args.checkpoint_interval == 0
        ):
            save_counter(counter, args.checkpoint)
        if args.log_every > 0 and total_processed % args.log_every == 0:
            logger.info(
                "Processed %d samples | distinct tokens=%d",
                total_processed,
                len(counter),
            )

    if batch:
        token_ids = tokenize_batch(tokenizer, batch)
        if args.use_gpu_bincount and token_ids:
            assert torch_module is not None
            token_tensor = torch_module.tensor(
                token_ids, device=torch_device, dtype=torch_module.int64
            )
            bincount = torch_module.bincount(
                token_tensor,
                minlength=tokenizer.vocab_size,
            )
            nonzero = (bincount > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
            for tid in nonzero:
                counter[tid] += int(bincount[tid].item())
        else:
            counter.update(token_ids)
        batch.clear()

    save_counter(counter, args.checkpoint)
    write_frequency_file(counter, args.freq_output)
    write_vocab_file(counter, args.vocab_output, args.topk, tokenizer.vocab_size)
    logger.info(
        "Done. Processed %d samples. Frequency file: %s | Vocab file: %s",
        total_processed,
        args.freq_output,
        args.vocab_output,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
