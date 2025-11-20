#!/usr/bin/env python3
"""
验证动态词汇表数据流是否完整的脚本
检查从 API 层到 ForwardBatch 的完整数据流
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

def check_dataflow():
    """检查数据流是否完整"""
    print("=" * 60)
    print("检查动态词汇表数据流完整性")
    print("=" * 60)
    
    issues = []
    checks = []
    
    # 1. 检查 ScheduleBatch.get_model_worker_batch() 是否提取 dynamic_vocab_token_ids
    print("\n1. 检查 ScheduleBatch.get_model_worker_batch()...")
    try:
        with open("python/sglang/srt/managers/schedule_batch.py", "r") as f:
            content = f.read()
            if "dynamic_vocab_token_ids" in content and "first_req.sampling_params.dynamic_vocab_token_ids" in content:
                checks.append("✅ ScheduleBatch.get_model_worker_batch() 提取 dynamic_vocab_token_ids")
                print("   ✅ 找到提取逻辑")
            else:
                issues.append("❌ ScheduleBatch.get_model_worker_batch() 未找到提取 dynamic_vocab_token_ids 的代码")
                print("   ❌ 未找到提取逻辑")
    except Exception as e:
        issues.append(f"❌ 无法读取 schedule_batch.py: {e}")
        print(f"   ❌ 错误: {e}")
    
    # 2. 检查 ForwardBatch.init_new() 是否设置 dynamic_vocab_token_ids
    print("\n2. 检查 ForwardBatch.init_new()...")
    try:
        with open("python/sglang/srt/model_executor/forward_batch_info.py", "r") as f:
            content = f.read()
            if "batch.dynamic_vocab_token_ids" in content and "ret.dynamic_vocab_token_ids" in content:
                checks.append("✅ ForwardBatch.init_new() 设置 dynamic_vocab_token_ids")
                print("   ✅ 找到设置逻辑")
            else:
                issues.append("❌ ForwardBatch.init_new() 未找到设置 dynamic_vocab_token_ids 的代码")
                print("   ❌ 未找到设置逻辑")
    except Exception as e:
        issues.append(f"❌ 无法读取 forward_batch_info.py: {e}")
        print(f"   ❌ 错误: {e}")
    
    # 3. 检查 LogitsProcessor 是否使用 dynamic_vocab_token_ids
    print("\n3. 检查 LogitsProcessor._project_hidden_to_vocab()...")
    try:
        with open("python/sglang/srt/layers/logits_processor.py", "r") as f:
            content = f.read()
            if "logits_metadata.dynamic_vocab_token_ids" in content and "dynamic_ids is not None" in content:
                checks.append("✅ LogitsProcessor 使用 dynamic_vocab_token_ids")
                print("   ✅ 找到使用逻辑")
            else:
                issues.append("❌ LogitsProcessor 未找到使用 dynamic_vocab_token_ids 的代码")
                print("   ❌ 未找到使用逻辑")
    except Exception as e:
        issues.append(f"❌ 无法读取 logits_processor.py: {e}")
        print(f"   ❌ 错误: {e}")
    
    # 4. 检查 API 层是否传递 dynamic_vocab_token_ids
    print("\n4. 检查 API 层...")
    try:
        with open("python/sglang/api/api_server.py", "r") as f:
            content = f.read()
            if "dynamic_vocab_token_ids" in content and "sampling_kwargs" in content:
                checks.append("✅ API 层传递 dynamic_vocab_token_ids")
                print("   ✅ 找到传递逻辑")
            else:
                issues.append("❌ API 层未找到传递 dynamic_vocab_token_ids 的代码")
                print("   ❌ 未找到传递逻辑")
    except Exception as e:
        issues.append(f"❌ 无法读取 api_server.py: {e}")
        print(f"   ❌ 错误: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)
    print(f"\n✅ 通过的检查: {len(checks)}")
    for check in checks:
        print(f"   {check}")
    
    if issues:
        print(f"\n❌ 发现的问题: {len(issues)}")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n✅ 所有检查通过！数据流看起来是完整的。")
        return True

def check_code_snippets():
    """检查关键代码片段"""
    print("\n" + "=" * 60)
    print("关键代码片段检查")
    print("=" * 60)
    
    # 检查 ScheduleBatch
    print("\n[ScheduleBatch.get_model_worker_batch()]")
    try:
        with open("python/sglang/srt/managers/schedule_batch.py", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "Extract dynamic_vocab_token_ids" in line or "dynamic_vocab_token_ids = None" in line:
                    print(f"   行 {i+1}: {line.strip()}")
                    # 打印后续几行
                    for j in range(i+1, min(i+15, len(lines))):
                        if lines[j].strip() and not lines[j].strip().startswith("#"):
                            print(f"   行 {j+1}: {lines[j].rstrip()}")
                            if "dynamic_vocab_token_ids=" in lines[j]:
                                break
                    break
    except Exception as e:
        print(f"   错误: {e}")
    
    # 检查 ForwardBatch
    print("\n[ForwardBatch.init_new()]")
    try:
        with open("python/sglang/srt/model_executor/forward_batch_info.py", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "Extract dynamic_vocab_token_ids" in line or ("batch.dynamic_vocab_token_ids" in line and "if" in line):
                    print(f"   行 {i+1}: {line.strip()}")
                    # 打印后续几行
                    for j in range(i+1, min(i+5, len(lines))):
                        print(f"   行 {j+1}: {lines[j].rstrip()}")
                    break
    except Exception as e:
        print(f"   错误: {e}")

if __name__ == "__main__":
    success = check_dataflow()
    check_code_snippets()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 数据流检查通过！动态词汇表功能应该已经启用。")
        print("\n建议：运行实际的端到端测试来验证功能是否正常工作。")
    else:
        print("❌ 发现潜在问题，请检查上述错误。")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

