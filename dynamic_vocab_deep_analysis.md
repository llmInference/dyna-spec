# 动态词汇表深度分析

## 当前状态

✅ **成功的部分**：
1. DynamicVocabManager 已创建（capacity=1024）
2. 词汇成功添加（populated_size=5: token IDs 100, 200, 300, 400, 500）
3. 权重成功提取（norms ~1.5, means ~0）
4. inverse_map 已更新（token_id -> reduced_idx映射）
5. slots范围已修正（只包含populated部分）

❌ **问题**：
- 草稿模型仍然只生成 token ID 1
- validation_log显示 `draft_generated_original_ids: [1, 1, 1]`

## 根本原因分析

### 假设1: Logits分数问题

**可能原因**：动态词汇的 logits 分数远低于静态词汇

**验证方法**：
1. 检查调试日志中的 `static_max` vs `dynamic_max`
2. 如果 dynamic_max << static_max，说明动态词汇的logits太低

**为什么会这样**：
- Token IDs 100, 200, 300等可能是低频词
- 对应的lm_head权重可能导致较低的logits
- 在softmax后，这些词的概率会非常小

### 假设2: 采样参数问题

**可能原因**：temperature, top_k, top_p等参数限制了动态词汇被选中

**当前测试**：
- temperature=0.8（中等）
- 需要测试更高的temperature（如1.5或2.0）

### 假设3: 模型语义不匹配

**根本问题**：这是一个架构性问题

**解释**：
- 静态词汇只有1个token（ID 1，可能是 `"`）
- 草稿模型在训练时学习的是"在任何上下文中都输出 `"` "
- 现在我们添加了token IDs 100-500，但模型从未见过这些token在这个reduced vocab中的使用
- **即使logits包含这些token，模型也不会知道何时应该使用它们**

这就像：
- 训练一个模型，词汇表只有"the"
- 然后运行时添加"cat", "dog", "bird"
- 模型仍然会一直输出"the"，因为它不知道何时该用其他词

## 解决方案

### 短期方案：验证机制是否工作

即使动态词汇在语义上不会被选择，我们也应该验证机制是否正常：

1. **添加高频词**：而不是100, 200, 300这些可能的低频词
2. **检查logits**：添加日志查看动态logits的实际分数
3. **强制采样**：使用极高的temperature + 修改logits来强制采样动态词汇

### 长期方案：正确使用动态词汇表

根据 README_Dyna.md，动态词汇表应该：
1. 用于添加高频词汇以提升性能
2. 静态词汇表应该包含合理数量的高频词（不只是1个）

**当前问题**：
- `numbers.txt` 只包含token ID 1
- 这不是一个realistic的使用场景
- 应该使用真正的高频词列表

## 下一步行动

1. **添加高频词测试**：
   ```bash
   # 获取Qwen词汇表前100个高频词的IDs
   # 添加到静态词汇表
   ```

2. **验证logits**：
   - 查看日志中的 dynamic_max vs static_max
   - 确认权重正确应用

3. **文档说明**：
   - 向用户解释当前配置（只有1个静态词汇）不是realistic场景
   - 建议正确的静态+动态词汇表配置

## 技术细节

### Logits计算流程

```python
# 1. 静态词汇logits（只有1个token）
static_logits = lm_head(hidden) @ static_weights.T  # shape: [batch, 1]

# 2. 动态词汇logits（5个tokens）
dynamic_logits = hidden @ dynamic_weights.T  # shape: [batch, 5]

# 3. 合并
combined_logits = torch.cat([static_logits, dynamic_logits], dim=1)  # shape: [batch, 6]

# 4. softmax
probs = softmax(combined_logits / temperature)  # shape: [batch, 6]

# 5. 采样
sampled_idx = sample(probs)  # 0-5

# 6. 映射回原始token ID
token_id = active_static_indices[sampled_idx]
# 如果 sampled_idx=0 -> token_id=1
# 如果 sampled_idx=1 -> token_id=100
# 如果 sampled_idx=2 -> token_id=200
# ...
```

### 预期行为

如果一切正常，即使语义不匹配，在高temperature下也应该偶尔采样到动态词汇。

**如果从未采样到**：说明 logits 分数差距太大，或者有其他bug。
