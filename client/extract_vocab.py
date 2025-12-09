#!/usr/bin/env python3
"""
脚本用于从 qwen_vocab_top32k_decoded(1).txt 文件中按顺序提取前 n 个词汇的 ID，
并以 vocab_400.json 的格式保存为 JSON 文件。
"""

import json
import sys
import argparse


def extract_vocab_ids(input_file, n, output_file):
    """
    从输入文件中提取前 n 个词汇的 ID，并保存为 JSON 格式。
    
    Args:
        input_file: 输入文件路径
        n: 要提取的词汇数量
        output_file: 输出 JSON 文件路径
    """
    vocab_ids = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 跳过表头和分隔线（前两行）
            next(f)  # 跳过 "ID        	Decoded_String"
            next(f)  # 跳过 "--------------------------------------------------"
            
            # 读取前 n 个词汇
            for i, line in enumerate(f):
                if i >= n:
                    break
                
                # 分割每行，提取 ID（第一列）
                parts = line.strip().split('\t')
                if parts and parts[0]:
                    try:
                        vocab_id = int(parts[0].strip())
                        vocab_ids.append(vocab_id)
                    except ValueError:
                        print(f"警告: 无法解析第 {i+3} 行的 ID: {parts[0]}")
                        continue
        
        # 保存为 JSON 格式
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_ids, f, indent=2, ensure_ascii=False)
        
        print(f"成功提取 {len(vocab_ids)} 个词汇 ID，已保存到 {output_file}")
        
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='从词汇文件中提取前 n 个词汇的 ID 并保存为 JSON 格式'
    )
    parser.add_argument(
        '-n', '--num',
        type=int,
        required=True,
        help='要提取的词汇数量'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='qwen_vocab_top32k_decoded(1).txt',
        help='输入文件路径（默认: qwen_vocab_top32k_decoded(1).txt）'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出 JSON 文件路径（默认: vocab_{n}.json）'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，使用默认名称
    if args.output is None:
        args.output = f'vocab_{args.num}.json'
    
    extract_vocab_ids(args.input, args.num, args.output)


if __name__ == '__main__':
    main()

