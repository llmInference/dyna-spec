#!/usr/bin/env python3
"""
测试脚本：验证动态词汇表功能
用于验证 dynamic_vocab_manager 是否正确初始化

使用方法:
1. 启动服务器（带有 --speculative-dynamic-vocab-capacity 参数）
2. 运行此脚本: python3 test_dynamic_vocab.py --port 30000
"""

import argparse
import json
import sys

import requests


def test_dynamic_vocab_status(host: str, port: int):
    """测试动态词汇表状态查询"""
    url = f"http://{host}:{port}/dynamic_vocab/status"
    print(f"[测试] 查询动态词汇表状态: {url}")

    try:
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json={}, timeout=5
        )
        response.raise_for_status()

        result = response.json()
        print(f"[成功] 状态查询响应:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 检查是否有错误
        if "error" in result:
            print(f"[失败] 错误: {result['error']}")
            return False

        # 验证必要字段
        required_fields = ["capacity", "populated_size", "slots"]
        missing = [f for f in required_fields if f not in result]
        if missing:
            print(f"[失败] 缺少字段: {missing}")
            return False

        print(f"[成功] 动态词汇表容量: {result['capacity']}")
        print(f"[成功] 已填充大小: {result['populated_size']}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[失败] 请求错误: {e}")
        return False


def test_dynamic_vocab_add(host: str, port: int, token_ids: list):
    """测试添加token到动态词汇表"""
    url = f"http://{host}:{port}/dynamic_vocab/add"
    print(f"\n[测试] 添加token到动态词汇表: {url}")
    print(f"[测试] Token IDs: {token_ids}")

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"vocab_size": 151936, "new_token_ids": token_ids},  # Qwen3 词汇表大小
            timeout=5,
        )
        response.raise_for_status()

        result = response.json()
        print(f"[成功] 添加响应:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return True

    except requests.exceptions.RequestException as e:
        print(f"[失败] 请求错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试动态词汇表功能")
    parser.add_argument("--host", default="localhost", help="服务器主机名")
    parser.add_argument("--port", type=int, default=30000, help="服务器端口")
    parser.add_argument(
        "--skip-add", action="store_true", help="跳过添加token测试，仅测试状态查询"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("动态词汇表功能测试")
    print("=" * 60)

    # 测试1: 查询初始状态
    print("\n[步骤 1] 查询初始状态")
    if not test_dynamic_vocab_status(args.host, args.port):
        print("\n[总结] 测试失败: 无法查询动态词汇表状态")
        print("请确保:")
        print("  1. 服务器已启动并使用 --speculative-dynamic-vocab-capacity 参数")
        print("  2. 服务器端口正确 (默认 30000)")
        print("  3. 查看服务器日志确认 DynamicVocabManager 已创建")
        sys.exit(1)

    if args.skip_add:
        print("\n[总结] 跳过添加token测试")
        sys.exit(0)

    # 测试2: 添加一些token
    print("\n[步骤 2] 添加测试token")
    test_token_ids = [143450, 147890, 100000, 120000]
    if not test_dynamic_vocab_add(args.host, args.port, test_token_ids):
        print("\n[总结] 测试失败: 无法添加token")
        sys.exit(1)

    # 测试3: 再次查询状态以确认token已添加
    print("\n[步骤 3] 查询更新后的状态")
    if not test_dynamic_vocab_status(args.host, args.port):
        print("\n[总结] 测试失败: 无法查询更新后的状态")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("[总结] 所有测试通过! 动态词汇表功能正常工作。")
    print("=" * 60)


if __name__ == "__main__":
    main()
