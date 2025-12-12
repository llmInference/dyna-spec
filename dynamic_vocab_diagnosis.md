# 动态词汇表未生效问题诊断指南

## 问题现象

✅ DynamicVocabManager 已成功创建（从日志确认）
✅ API `/dynamic_vocab/add` 调用成功
❌ 草稿模型只生成 token ID 1（静态词汇表中唯一的词）
❌ 动态添加的词汇未在采样中出现

## 诊断步骤

### 步骤1: 检查实际添加了多少词

```bash
# 启动服务器
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path Qwen/Qwen3-1.7B \
  --speculative-dynamic-vocab-capacity 1024 \
  --speculative-use-static-vocab \
  --speculative-static-vocab-path ./numbers.txt \
  --log-level info

# 查询初始状态（应该 populated_size = 0）
curl -X POST http://localhost:30000/dynamic_vocab/status \
  -H "Content-Type: application/json" \
  -d '{}'

# 添加测试词汇
curl -X POST http://localhost:30000/dynamic_vocab/add \
  -H "Content-Type: application/json" \
  -d '{
    "vocab_size": 151936,
    "new_token_ids": [100, 200, 300, 400, 500, 1000, 2000, 3000]
  }'

# 再次查询状态（应该 populated_size = 8）
curl -X POST http://localhost:30000/dynamic_vocab/status \
  -H "Content-Type: application/json" \
  -d '{}'
```

**预期结果**:
```json
{
  "capacity": 1024,
  "populated_size": 8,
  "slots": [100, 200, 300, 400, 500, 1000, 2000, 3000, -1, -1, ...],
  "weights_head_norm": [...],
  "weights_head_mean": [...]
}
```

### 步骤2: 检查词汇表文件内容

```bash
# 检查静态词汇表
cat numbers.txt
# 应该只包含: 1

# 检查动态词汇表文件（如果使用文件）
head -20 qwen_vocab_top32k_ids_sorted.txt
```

### 步骤3: 测试推理

```bash
curl http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "sampling_params": {
      "max_new_tokens": 30,
      "temperature": 0.8
    }
  }'
```

检查服务器日志，查找：
- `accept_rate`: 如果一直很低（如 0.25），说明草稿模型生成的词大多被拒绝
- 推理过程中是否有关于动态词汇表的日志

## 可能的问题和解决方案

### 问题1: 文件中的token都在静态词汇表中

**症状**: 日志显示 "skip token X already in static vocab"，但 `populated_size` 仍为 0

**原因**: `qwen_vocab_top32k_ids_sorted.txt` 中所有token都已存在于 `numbers.txt`

**解决**:
```bash
# 使用明确不同的token IDs测试
curl -X POST http://localhost:30000/dynamic_vocab/add \
  -H "Content-Type: application/json" \
  -d '{
    "vocab_size": 151936,
    "new_token_ids": [10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
  }'
```

### 问题2: 权重未正确提取或更新

**检查**: 查看日志中是否有 `weights_head_norm` 和 `weights_head_mean`

如果这些值都是0或非常小，说明权重提取有问题。

**可能原因**:
- `lm_head` 权重访问失败
- 张量并行(TP)环境下权重分片导致无法提取

### 问题3: 采样器设置导致动态词汇不被选中

即使动态词汇表中有词，如果其logits分数很低，仍然不会被采样。

**测试**: 使用更高的temperature和top_k

```bash
curl http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test",
    "sampling_params": {
      "max_new_tokens": 20,
      "temperature": 1.5,
      "top_k": 100,
      "top_p": 0.99
    }
  }'
```

### 问题4: LogitsProcessor未正确合并动态logits

查看 `logits_processor.py` 第 1001-1024 行的代码，动态logits应该被拼接到静态logits后面。

**验证**: 启用更详细的日志

修改 `logits_processor.py`:
```python
# 在第 1018 行 torch.cat 之后添加
logger.info(f"Static logits shape: {logits.shape}, Dynamic logits shape: {dynamic_logits.shape}")
logger.info(f"Populated size: {self.dynamic_vocab_manager.populated_size}")
logger.info(f"Dynamic logits sample: {dynamic_logits[0, :10]}")
```

## 推荐的完整测试流程

```bash
# 1. 启动服务器（终端1）
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path Qwen/Qwen3-1.7B \
  --speculative-dynamic-vocab-capacity 1024 \
  --speculative-use-static-vocab \
  --speculative-static-vocab-path ./numbers.txt \
  --log-level info

# 2. 在另一个终端执行测试
# 查询初始状态
curl -s -X POST http://localhost:30000/dynamic_vocab/status \
  -H "Content-Type: application/json" \
  -d '{}' | jq '.'

# 添加少量测试token
curl -s -X POST http://localhost:30000/dynamic_vocab/add \
  -H "Content-Type: application/json" \
  -d '{
    "vocab_size": 151936,
    "new_token_ids": [100, 200, 300]
  }' | jq '.'

# 确认添加成功
curl -s -X POST http://localhost:30000/dynamic_vocab/status \
  -H "Content-Type: application/json" \
  -d '{}' | jq '.'

# 推理测试
curl -s http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test",
    "sampling_params": {
      "max_new_tokens": 20,
      "temperature": 1.0
    }
  }' | jq '.output_ids'
```

## 预期行为

如果一切正常：
1. `populated_size` 应该等于添加的token数量
2. `weights_head_norm` 应该有非零值
3. 生成的output_ids中可能（但不一定）包含添加的token IDs

**注意**: 即使动态词汇表正常工作，由于采样的随机性，不一定每次都会采样到新添加的词。关键是要确保：
- `populated_size > 0`
- `weights_head_norm` 有合理的值（不是全0）
- logits正确合并

## 下一步建议

如果以上测试都正常，但仍然观察不到动态词汇被使用，可能需要：

1. **增加动态词汇表大小**: 添加更多高频词
2. **检查权重质量**: 确保提取的权重与原始lm_head权重一致
3. **调整采样参数**: 使用更高的temperature和top_k以增加采样多样性
4. **查看验证日志**: 分析 `validation_log` 中 `draft_generated_original_ids` 的分布
