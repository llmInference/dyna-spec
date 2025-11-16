#!/usr/bin/env python3
"""Utility to launch multiple shards of stream_slimpajama_vocab.py and merge their outputs."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

logger = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple OpenWebText shards in parallel and merge results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-shards", type=int, default=4, help="Number of parallel shards"
    )
    parser.add_argument(
        "--script",
        default="reducedVocab/stream_slimpajama_vocab.py",
        help="Path to the streaming script",
    )
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=32000)
    parser.add_argument("--checkpoint-interval", type=int, default=200000)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--work-dir", default="/tmp")
    parser.add_argument("--use-gpu-bincount", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--freq-output", default="token_freq.part{idx}.txt")
    parser.add_argument("--vocab-output", default="custom_static_vocab.part{idx}.txt")
    parser.add_argument("--checkpoint", default="slim_counter.part{idx}.pkl")
    parser.add_argument("--text-key", default="text")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Run merge_token_freq.py after shards finish",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, idx: int) -> List[str]:
    freq_path = os.path.join(args.work_dir, args.freq_output.format(idx=idx))
    vocab_path = os.path.join(args.work_dir, args.vocab_output.format(idx=idx))
    ckpt_path = os.path.join(args.work_dir, args.checkpoint.format(idx=idx))
    log_file = Path(args.log_dir) / f"slim_vocab.part{idx}.log"
    cmd = [
        sys.executable,
        args.script,
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--tokenizer",
        args.tokenizer,
        "--freq-output",
        freq_path,
        "--vocab-output",
        vocab_path,
        "--topk",
        str(args.topk),
        "--batch-size",
        str(args.batch_size),
        "--checkpoint",
        ckpt_path,
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--text-key",
        args.text_key,
        "--shard-total",
        str(args.num_shards),
        "--shard-index",
        str(idx),
    ]
    if args.use_gpu_bincount:
        cmd.append("--use-gpu-bincount")
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    return cmd, log_file


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    processes = []
    for idx in range(args.num_shards):
        cmd, log_file = build_command(args, idx)
        with open(log_file, "wb") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            processes.append((proc, log_file))
            print(f"Started shard {idx}: PID={proc.pid}, log={log_file}")
    for proc, log_path in processes:
        proc.wait()
        print(f"Shard log: {log_path} exit={proc.returncode}")
    if args.merge:
        merge_cmd = [
            sys.executable,
            "reducedVocab/merge_token_freq.py",
            "--output",
            os.path.join(args.work_dir, "token_freq.all.txt"),
            "--vocab-output",
            os.path.join(args.work_dir, "custom_static_vocab.txt"),
            "--topk",
            str(args.topk),
            "--tokenizer",
            args.tokenizer,
        ]
        for idx in range(args.num_shards):
            merge_cmd.extend(
                [
                    "--inputs",
                    os.path.join(args.work_dir, args.freq_output.format(idx=idx)),
                ]
            )
        subprocess.check_call(merge_cmd)
        print("Merged outputs into final vocab.")


if __name__ == "__main__":
    main()
