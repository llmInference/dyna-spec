# SlimPajama Static Vocabulary Toolkit

This folder hosts scripts that implement the requirement documented in `README_Dyna.md`:

> "基于大语料按出现率生成静态词汇表" —— stream a large corpus (default: `Skylion007/openwebtext`), tokenize
> with the draft model tokenizer, count token frequencies, and emit
> (1) a frequency report and (2) a static vocab file sorted by token id.

## Components

| File | Description |
| --- | --- |
| `stream_slimpajama_vocab.py` | Streams SlimPajama, tokenizes with a user-provided tokenizer, maintains token frequency counts, and writes both a frequency file and a sorted static vocabulary (top-k token IDs). Supports dataset sharding, checkpoint resume, and optional GPU-accelerated `torch.bincount`. |
| `merge_token_freq.py` | Utility that merges multiple frequency files (e.g., from parallel runs) and optionally emits a final static vocab file. |

## Quick Start

```bash
pip install zstandard
python reducedVocab/stream_slimpajama_vocab.py \
  --tokenizer Qwen/Qwen3-4B \
  --freq-output /tmp/token_freq.txt \
  --vocab-output /tmp/custom_static_vocab.txt \
  --topk 32000 \
  --batch-size 64 \
  --checkpoint /tmp/slimpajama_counter.pkl
```

By default the script streams `Skylion007/openwebtext` so you only need to pass the dataset flag if you plan to work on another corpus.
Because OpenWebText is a community dataset, the script automatically sets `--trust-dataset-code` to `true`; use `--no-trust-dataset-code` if you prefer to run only first-party datasets.

The script:

1. Streams SlimPajama via `datasets.load_dataset(..., streaming=True)` to avoid storing the full corpus locally.
2. Tokenizes batches of records using the specified tokenizer (matching the draft model).
3. Updates a global `Counter` and, optionally, checkpoints it to disk for resuming.
4. Writes:
   - `token_freq.txt`: `token_id\tcount`, sorted by frequency (desc) then token id (asc).
   - `custom_static_vocab.txt`: top-k token IDs filtered to valid vocab range and sorted ascending, one per line.

### Parallelizing across shards

To saturate bandwidth or multiple GPUs, launch several instances of the script, each processing a dataset shard via `--shard-total` / `--shard-index`. Example for four parallel workers:

```bash
TOKENIZER=Qwen/Qwen3-4B
for IDX in 0 1 2 3; do
  python reducedVocab/stream_slimpajama_vocab.py \
    --dataset Skylion007/openwebtext \
    --split train \
    --tokenizer "$TOKENIZER" \
    --freq-output /tmp/token_freq.part${IDX}.txt \
    --vocab-output /tmp/custom_static_vocab.part${IDX}.txt \
    --topk 32000 \
    --batch-size 512 \
    --checkpoint /tmp/slim_counter.part${IDX}.pkl \
    --shard-total 4 \
    --shard-index ${IDX} \
    --use-gpu-bincount \
    --checkpoint-interval 200000 \
    --log-every 20000 \
    > logs/shard_${IDX}.log 2>&1 &
done
wait
```

Each worker handles disjoint samples, so the union matches the full dataset (bounded by available bandwidth). Increase `--batch-size` to reduce tokenizer overhead; `--use-gpu-bincount` offloads per-batch counting to CUDA (requires PyTorch + GPU).

### Merging partial outputs

After parallel runs finish, merge the partial frequency files and create the final vocab:

```bash
python reducedVocab/merge_token_freq.py \
  --inputs /tmp/token_freq.part*.txt \
  --output /tmp/token_freq.all.txt \
  --vocab-output /tmp/custom_static_vocab.txt \
  --topk 32000 \
  --tokenizer Qwen/Qwen3-4B
```

`merge_token_freq.py` aggregates counts (summing duplicate token IDs), preserves the descending frequency order, and optionally writes the final static vocab (ascending token IDs) when tokenizer info is provided.

### Automated shard orchestration

If you prefer not to manage background jobs manually, `run_parallel_slimpajama.py` launches the desired number of shards, streams logs per worker, and optionally runs the merge helper once every shard completes. Supply the tokenizer, optional GPU counting flag, shard count, and working directory where outputs and checkpoints will land:

```bash
python reducedVocab/run_parallel_slimpajama.py \
  --tokenizer Qwen/Qwen3-4B \
  --work-dir /tmp/slimpajama \
  --num-shards 6 \
  --use-gpu-bincount \
  --merge
```

The script writes one log per shard inside `--log-dir` (default `logs/`), names frequency/vocab/checkpoint files with the shard index, and runs `merge_token_freq.py` to produce the consolidated outputs if `--merge` is supplied.
It defaults to `Skylion007/openwebtext`, so the example above already streams OpenWebText unless you override `--dataset`.

## Notes

- Requires `datasets`, `transformers`, `tqdm`, and `zstandard` packages. `torch` is needed only when `--use-gpu-bincount` is enabled.
- Use `--text-key` to point at nested text fields (dot notation supported) if the dataset schema differs.
- Use `--max-samples` for smoke testing and `--checkpoint-interval` to persist progress regularly.
- When running many shards, give each worker a unique checkpoint path (e.g., `/tmp/slim_counter.partX.pkl`).
- If you want the tokenizer to leverage Rust multi-threading, export `TOKENIZERS_PARALLELISM=true` before launching the scripts.
- The resulting vocab file is ready for `--speculative-static-vocab-path` or `custom_vocab_path` in SGlang.
