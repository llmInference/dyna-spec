#!/usr/bin/env python3
"""生成非连续的词汇表 JSON 文件

从 qwen3-4b 的词汇表中随机选择指定数量的 token ID，生成符合动态词汇表格式的 JSON 文件。

示例:
    python client/generate_non_continuous_vocab.py --count 400 --output vocab_400.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

from transformers import AutoTokenizer


def load_qwen_tokenizer(model_path: str = "Qwen/Qwen3-4B") -> Any:
    """加载 Qwen3-4B 的 tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"无法加载 tokenizer: {e}")


def generate_non_continuous_vocab(
    tokenizer: Any,
    count: int,
    seed: Optional[int] = None,
) -> list[int]:
    """从词汇表中随机选择指定数量的 token ID（非连续）"""
    vocab_size = tokenizer.vocab_size
    
    if count > vocab_size:
        raise ValueError(
            f"请求的 token 数量 ({count}) 超过了词汇表大小 ({vocab_size})"
        )
    
    # 设置随机种子（如果提供）
    if seed is not None:
        random.seed(seed)
    
    # 从所有有效的 token ID 中随机选择
    all_token_ids = list(range(vocab_size))
    selected_token_ids = random.sample(all_token_ids, count)
    
    # 排序以便于查看（可选，但保持顺序可能更有用）
    selected_token_ids.sort()
    
    return selected_token_ids


def save_vocab_json(
    token_ids: list[int],
    output_path: Path,
    format_type: str = "array",
    indent: int = 2,
) -> None:
    """保存词汇表为 JSON 文件
    
    Args:
        token_ids: token ID 列表
        output_path: 输出文件路径
        format_type: 格式类型，"array" 或 "object"
        indent: JSON 缩进级别
    """
    if format_type == "array":
        # 数组格式: [0, 1, 2, ...]
        payload = token_ids
    elif format_type == "object":
        # 对象格式: {"token_ids": [0, 1, 2, ...]}
        payload = {"token_ids": token_ids}
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
    
    print(f"已生成 {len(token_ids)} 个 token ID 到 {output_path}")
    print(f"格式: {format_type}")
    print(f"Token ID 范围: {min(token_ids)} - {max(token_ids)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Qwen3-4B 词汇表中生成非连续的词汇表 JSON 文件"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=400,
        help="要选择的 token 数量（默认: 400）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/llminference/syq/dyna-spec/client/vocab_400.json"),
        help="输出 JSON 文件路径（默认:/home/llminference/syq/dyna-spec/client/vocab_400.json）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="模型路径（默认: Qwen/Qwen3-4B）",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["array", "object"],
        default="array",
        help="输出格式：'array' 为 [0,1,2,...]，'object' 为 {\"token_ids\": [0,1,2,...]}（默认: array）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选，用于可重复的结果）",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON 缩进级别（默认: 2）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"正在加载 tokenizer: {args.model_path}")
    tokenizer = load_qwen_tokenizer(args.model_path)
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    print(f"正在随机选择 {args.count} 个 token ID...")
    token_ids = generate_non_continuous_vocab(
        tokenizer, args.count, seed=args.seed
    )
    
    save_vocab_json(
        token_ids,
        args.output,
        format_type=args.format,
        indent=args.indent,
    )


if __name__ == "__main__":
    main()

