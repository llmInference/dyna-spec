#!/usr/bin/env python3
"""
分析推测解码输出脚本

功能：
1. 整理草稿模型每轮次的输出，直观显示
2. 整理目标模型的输出，直观显示
3. 将结果保存到markdown文件

使用方法：
    python analyze_spec_output.py < input.txt
    或者
    cat output.txt | python analyze_spec_output.py
    或者
    python analyze_spec_output.py input.txt -o output.md
"""

import json
import sys
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime


def parse_draft_tokens(draft_tokens: List[Dict]) -> str:
    """将草稿模型的token列表转换为字符串"""
    tokens = []
    for token_info in draft_tokens:
        token = token_info.get('token', '')
        tokens.append(token)
    return ''.join(tokens)


def extract_target_tokens(round_detail: Dict) -> List[str]:
    """从验证结果中提取目标模型的token"""
    target_tokens = []
    verification_results = round_detail.get('verification_results', [])
    
    # 构建位置到验证结果的映射
    verification_map = {}
    for result in verification_results:
        position = result.get('position', -1)
        verification_map[position] = result
    
    # 按位置顺序构建目标模型的输出
    # 如果被接受，使用草稿模型的token（因为被目标模型接受了）
    # 如果被拒绝，使用目标模型实际选择的token
    draft_tokens = round_detail.get('draft_tokens', [])
    for i, token_info in enumerate(draft_tokens):
        if i in verification_map:
            result = verification_map[i]
            accepted = result.get('accepted', False)
            if accepted:
                # 被接受的token，目标模型也选择了这个
                token = token_info.get('token', '')
                target_tokens.append(token)
            else:
                # 被拒绝的token，使用目标模型实际选择的token
                target_token = result.get('target_token')
                if target_token is not None and target_token != '':
                    target_tokens.append(target_token)
    
    return target_tokens


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """生成markdown格式的分析报告"""
    lines = []
    
    # 标题
    lines.append("# 推测解码输出分析报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 基本信息
    text = data.get('text', '')
    draft_avg_accept_length = data.get('draft_avg_accept_length', 0.0)
    spec_total_rounds = data.get('spec_total_rounds', 0)
    
    lines.append("## 基本信息")
    lines.append("")
    lines.append(f"- **最终生成文本**: `{text}`")
    lines.append(f"- **平均接受长度**: {draft_avg_accept_length:.2f}")
    lines.append(f"- **总轮数**: {spec_total_rounds}")
    lines.append("")
    
    # 分析每轮详情
    spec_round_details = data.get('spec_round_details', [])
    
    # 草稿模型输出
    lines.append("## 草稿模型输出（按轮次）")
    lines.append("")
    
    for round_idx, round_detail in enumerate(spec_round_details, 1):
        draft_tokens = round_detail.get('draft_tokens', [])
        verification_results = round_detail.get('verification_results', [])
        accept_length = round_detail.get('accept_length', 0)
        
        # 构建位置到验证结果的映射
        verification_map = {}
        for result in verification_results:
            position = result.get('position', -1)
            verification_map[position] = result
        
        lines.append(f"### 轮次 {round_idx}")
        lines.append("")
        
        # 显示所有草稿token，标记哪些被接受
        draft_output = []
        for i, token_info in enumerate(draft_tokens):
            token = token_info.get('token', '')
            # 转义markdown特殊字符
            token_escaped = token.replace('|', '\\|').replace('`', '\\`')
            if i in verification_map:
                accepted = verification_map[i].get('accepted', False)
                if accepted:
                    draft_output.append(f"**{token_escaped}**")
                else:
                    draft_output.append(f"~~{token_escaped}~~")
            else:
                # 如果没有验证结果，默认标记为未接受
                draft_output.append(f"~~{token_escaped}~~")
        
        lines.append(f"**草稿模型输出**: {' '.join(draft_output)}")
        lines.append("")
        lines.append(f"- **接受长度**: {accept_length}/{len(draft_tokens)}")
        lines.append("")
        
        # 显示验证详情（包括接受和拒绝的）
        if verification_results:
            lines.append("**验证详情**:")
            lines.append("")
            lines.append("| 位置 | 草稿Token | 草稿Token ID | 目标Token | 目标Token ID | 状态 |")
            lines.append("|------|-----------|--------------|-----------|--------------|------|")
            for result in sorted(verification_results, key=lambda x: x.get('position', -1)):
                position = result.get('position', -1)
                draft_token = result.get('draft_token', '')
                draft_token_id = result.get('draft_token_id', -1)
                target_token = result.get('target_token')
                target_token_id = result.get('target_token_id', -1)
                accepted = result.get('accepted', False)
                
                draft_token_escaped = str(draft_token).replace('|', '\\|')
                if target_token is None:
                    target_token_escaped = '[None]'
                elif target_token == '':
                    target_token_escaped = '[空字符串]'
                else:
                    target_token_escaped = str(target_token).replace('|', '\\|')
                
                status = "✓ 接受" if accepted else "✗ 拒绝"
                lines.append(f"| {position} | `{draft_token_escaped}` | {draft_token_id} | `{target_token_escaped}` | {target_token_id} | {status} |")
            lines.append("")
    
    # 目标模型输出
    lines.append("## 目标模型输出（按轮次）")
    lines.append("")
    
    for round_idx, round_detail in enumerate(spec_round_details, 1):
        lines.append(f"### 轮次 {round_idx}")
        lines.append("")
        
        target_tokens = extract_target_tokens(round_detail)
        if target_tokens:
            target_output = ''.join(target_tokens)
            target_output_escaped = target_output.replace('`', '\\`')
            lines.append(f"**目标模型输出**: `{target_output_escaped}`")
        else:
            lines.append("**目标模型输出**: [无输出]")
        lines.append("")
        
        # 显示接受的token
        verification_results = round_detail.get('verification_results', [])
        draft_tokens = round_detail.get('draft_tokens', [])
        
        # 构建位置到验证结果的映射
        verification_map = {}
        for result in verification_results:
            position = result.get('position', -1)
            verification_map[position] = result
        
        accepted_tokens = []
        for i, token_info in enumerate(draft_tokens):
            if i in verification_map:
                if verification_map[i].get('accepted', False):
                    token = token_info.get('token', '')
                    accepted_tokens.append(token)
        
        if accepted_tokens:
            accepted_str = ''.join(accepted_tokens).replace('`', '\\`')
            lines.append(f"- **接受的token（草稿=目标）**: `{accepted_str}`")
            lines.append("")
        
        # 显示拒绝的token对比
        rejected_results = [r for r in verification_results if not r.get('accepted', False)]
        if rejected_results:
            lines.append("**拒绝的token对比**:")
            lines.append("")
            for result in sorted(rejected_results, key=lambda x: x.get('position', -1)):
                position = result.get('position', -1)
                draft_token = result.get('draft_token', '')
                target_token = result.get('target_token')
                
                draft_token_escaped = str(draft_token).replace('`', '\\`')
                if target_token is None:
                    lines.append(f"- 位置 {position}: 草稿 `{draft_token_escaped}` ≠ 目标 [None]")
                elif target_token == '':
                    lines.append(f"- 位置 {position}: 草稿 `{draft_token_escaped}` ≠ 目标 [空字符串]")
                else:
                    target_token_escaped = str(target_token).replace('`', '\\`')
                    lines.append(f"- 位置 {position}: 草稿 `{draft_token_escaped}` ≠ 目标 `{target_token_escaped}`")
            lines.append("")
    
    # 对比总结
    lines.append("## 对比总结")
    lines.append("")
    
    total_draft_tokens = 0
    total_accepted_tokens = 0
    total_target_tokens = 0
    
    for round_detail in spec_round_details:
        draft_tokens = round_detail.get('draft_tokens', [])
        accept_length = round_detail.get('accept_length', 0)
        target_tokens = extract_target_tokens(round_detail)
        
        total_draft_tokens += len(draft_tokens)
        total_accepted_tokens += accept_length
        total_target_tokens += len(target_tokens)
    
    accept_rate = (total_accepted_tokens / total_draft_tokens * 100) if total_draft_tokens > 0 else 0
    
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 草稿模型总token数 | {total_draft_tokens} |")
    lines.append(f"| 接受的token数 | {total_accepted_tokens} |")
    lines.append(f"| 目标模型总token数 | {total_target_tokens} |")
    lines.append(f"| 接受率 | {accept_rate:.2f}% |")
    lines.append("")
    
    return '\n'.join(lines)


def analyze_spec_output(data: Dict[str, Any], output_file: Optional[str] = None):
    """分析推测解码输出"""
    # 生成markdown报告
    markdown_content = generate_markdown_report(data)
    
    # 保存到文件
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"分析结果已保存到: {output_file}", file=sys.stderr)
        except Exception as e:
            print(f"保存文件失败: {e}", file=sys.stderr)
    
    # 同时打印到控制台（简化版本）
    print("=" * 80)
    print("推测解码输出分析")
    print("=" * 80)
    print()
    
    # 基本信息
    text = data.get('text', '')
    draft_avg_accept_length = data.get('draft_avg_accept_length', 0.0)
    spec_total_rounds = data.get('spec_total_rounds', 0)
    
    print(f"最终生成文本: {text}")
    print(f"平均接受长度: {draft_avg_accept_length:.2f}")
    print(f"总轮数: {spec_total_rounds}")
    print()
    
    if output_file:
        print(f"详细分析结果已保存到: {output_file}")
        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='分析推测解码输出',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analyze_spec_output.py input.txt
  python analyze_spec_output.py input.txt -o output.md
  cat output.txt | python analyze_spec_output.py -o output.md
        """
    )
    parser.add_argument('input_file', nargs='?', help='输入文件（JSON格式，每行一个JSON对象）')
    parser.add_argument('-o', '--output', dest='output_file', help='输出markdown文件路径')
    
    args = parser.parse_args()
    
    output_file = args.output_file
    all_data = []
    
    if args.input_file:
        # 从文件读取
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"解析JSON失败: {e}", file=sys.stderr)
                        print(f"问题行: {line[:100]}...", file=sys.stderr)
        except FileNotFoundError:
            print(f"错误: 文件 '{args.input_file}' 不存在", file=sys.stderr)
            sys.exit(1)
    else:
        # 从标准输入读取
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                all_data.append(data)
            except json.JSONDecodeError as e:
                print(f"解析JSON失败: {e}", file=sys.stderr)
                print(f"问题行: {line[:100]}...", file=sys.stderr)
    
    # 处理每个JSON对象
    if not all_data:
        print("错误: 没有找到有效的JSON数据", file=sys.stderr)
        sys.exit(1)
    
    # 如果指定了输出文件，合并所有数据到一个markdown文件
    if output_file and len(all_data) > 1:
        lines = []
        lines.append("# 推测解码输出分析报告（批量）")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**分析条目数**: {len(all_data)}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        for idx, data in enumerate(all_data, 1):
            lines.append(f"## 条目 {idx}")
            lines.append("")
            lines.append(generate_markdown_report(data))
            lines.append("")
            lines.append("---")
            lines.append("")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"分析结果已保存到: {output_file}", file=sys.stderr)
        except Exception as e:
            print(f"保存文件失败: {e}", file=sys.stderr)
        
        # 打印摘要
        for idx, data in enumerate(all_data, 1):
            print(f"\n条目 {idx}:")
            analyze_spec_output(data, None)
    else:
        # 单独处理每个数据
        for idx, data in enumerate(all_data, 1):
            if len(all_data) > 1:
                print(f"\n{'=' * 80}")
                print(f"条目 {idx}/{len(all_data)}")
                print(f"{'=' * 80}\n")
            
            if output_file and len(all_data) == 1:
                # 单个数据，直接保存
                analyze_spec_output(data, output_file)
            else:
                # 多个数据，为每个生成单独的文件
                if output_file:
                    base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
                    ext = output_file.rsplit('.', 1)[1] if '.' in output_file else 'md'
                    single_output = f"{base_name}_{idx}.{ext}"
                    analyze_spec_output(data, single_output)
                else:
                    analyze_spec_output(data, None)
            
            if idx < len(all_data):
                print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()

