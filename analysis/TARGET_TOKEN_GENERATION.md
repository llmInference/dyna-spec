# 目标模型在草稿模型每个位置的Token生成过程

## 概述

在 EAGLE 推测解码中，目标模型需要验证草稿模型生成的每个 token。目标模型对每个位置的预测 token 是通过以下步骤得到的：

## 完整流程

### 1. 准备验证输入 (`prepare_for_verify`)

**代码位置**: `eagle_info.py:106-150`

```python
def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
    # 关键步骤：将草稿模型生成的 tokens 设置为目标模型的输入
    batch.input_ids = self.draft_token  # 草稿模型生成的 tokens
    
    # 分配 KV cache 空间
    # ... KV cache 分配逻辑 ...
```

**说明**:
- `self.draft_token` 是草稿模型生成的 tokens，形状为 `(batch_size * draft_token_num,)`
- 这些 tokens 会被设置为目标模型的 `input_ids`
- 目标模型会基于这些输入（以及之前的 prompt，存储在 KV cache 中）进行前向传播

### 2. 目标模型前向传播 (`verify`)

**代码位置**: `eagle_worker.py:638-699`

```python
def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
    # 准备验证输入
    spec_info.prepare_for_verify(batch, self.page_size)
    
    # 设置前向模式为 TARGET_VERIFY
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    
    # 调用目标模型进行前向传播
    batch_result = self.target_worker.forward_batch_generation(
        model_worker_batch, is_verify=True
    )
    logits_output = batch_result.logits_output
```

**说明**:
- 目标模型接收草稿 tokens 作为输入
- 进行前向传播，生成 `logits_output.next_token_logits`
- `next_token_logits` 的形状是 `(batch_size * draft_token_num, vocab_size)`
- 每个位置的 logits 表示目标模型在该位置预测下一个 token 的概率分布

### 3. 从 Logits 提取目标预测 (`verify` 方法)

**代码位置**: `eagle_info.py:286-300` (Greedy 模式) 或 `eagle_info.py:301-352` (Non-Greedy 模式)

#### Greedy 模式（最常见）:

```python
# 从 logits 中取 argmax 得到目标模型的预测
target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
# 形状: (batch_size * draft_token_num,)
target_predict = target_predict.reshape(bs, self.draft_token_num)
# 形状: (batch_size, draft_token_num)
```

**说明**:
- `next_token_logits[i]` 是目标模型在位置 i 的 logits（概率分布）
- `argmax` 取概率最大的 token ID
- `target_predict[batch_idx, pos]` 是目标模型在 batch_idx 的 pos 位置预测的 token ID

#### Non-Greedy 模式:

```python
# 应用 temperature 和 top-k/top-p
target_probs = F.softmax(
    logits_output.next_token_logits / expanded_temperature, dim=-1
)
target_probs = top_k_renorm_prob(target_probs, ...)
target_probs = top_p_renorm_prob(target_probs, ...)

# 使用 rejection sampling 进行采样
tree_speculative_sampling_target_only(
    predicts=predict,  # 输出：目标模型的预测
    target_probs=target_probs,
    ...
)
```

### 4. 索引对应关系（关键！）

**重要**: `next_token_logits` 的索引对应关系存在偏移：

在 EAGLE 验证中，输入序列是：
```
[prompt_tokens (在 KV cache 中)..., draft_token_0, draft_token_1, draft_token_2, ...]
```

但是，`next_token_logits` 只针对草稿 tokens 计算：
- `next_token_logits[0]`: 目标模型在 **prompt 之后** 预测的 logits（应该预测 `draft_token_0`）
- `next_token_logits[1]`: 目标模型在 **draft_token_0 之后** 预测的 logits（应该预测 `draft_token_1`）
- `next_token_logits[2]`: 目标模型在 **draft_token_1 之后** 预测的 logits（应该预测 `draft_token_2`）

**但是**，根据 `verify_tree_greedy_native` 的代码分析：
```python
# 在 verify_tree_greedy_native 中：
comparison_result = candidates[:, 1] == target_predict[:, 0]
```

这说明：
- `target_predict[0]` 用于验证 `candidates[1]`（草稿模型的第2个token）
- `target_predict[1]` 用于验证 `candidates[2]`（草稿模型的第3个token）

**这意味着**：
- `target_predict[i]` 验证的是 `candidates[i+1]`
- 对于 `candidates[0]`，需要特殊处理

### 5. 验证比较逻辑 (`verify_tree_greedy_func`)

**代码位置**: `eagle_utils.py:345-383`

```python
def verify_tree_greedy_func(...):
    # 对于每个 batch
    for bx in range(batch_size):
        cur_candidates = candidates[bx]  # 草稿模型的 tokens
        cur_target = target_predict[bx]  # 目标模型的预测
        
        # 遍历草稿 tokens，比较是否匹配
        for node in tree_structure:
            draft_token = cur_candidates[node]
            target_token = cur_target[last_accepted_idx - num_draft_tokens * bx]
            
            if draft_token == target_token:
                # 接受这个 token
                accept_index[bx, num_accepted] = draft_idx
                num_accepted += 1
            else:
                # 拒绝，停止验证后续 tokens
                break
```

### 6. 收集验证结果

**代码位置**: `eagle_info.py:432-471`

```python
# 对于每个位置，收集验证结果
for pos in range(min(len(req_candidates), self.draft_token_num)):
    draft_token_id = int(req_candidates[pos])
    
    # 获取目标模型的预测
    if pos == 0:
        # 位置0需要特殊处理
        target_token_id = get_from_predict_or_fallback()
    else:
        # 位置 > 0: target_predict[pos-1] 验证 candidates[pos]
        target_idx = pos - 1
        target_token_id = int(req_target_predict[target_idx])
    
    # 检查是否被接受
    is_accepted = pos in accepted_positions
    
    # 添加到验证结果
    verification_results.append({
        "position": pos,
        "draft_token_id": draft_token_id,
        "target_token_id": target_token_id,
        "accepted": is_accepted
    })
```

## 关键代码位置总结

1. **准备输入**: `eagle_info.py:111` - `batch.input_ids = self.draft_token`
2. **目标模型前向传播**: `eagle_worker.py:661-663` - `target_worker.forward_batch_generation()`
3. **提取目标预测 (Greedy)**: `eagle_info.py:287-288` - `torch.argmax(logits_output.next_token_logits)`
4. **提取目标预测 (Non-Greedy)**: `eagle_info.py:307-337` - `tree_speculative_sampling_target_only()`
5. **验证比较**: `eagle_utils.py:345-383` - `verify_tree_greedy_func()`
6. **收集结果**: `eagle_info.py:432-471` - 收集每个位置的验证结果

## 数据流示例

假设草稿模型生成了 4 个 tokens: `[20, 0, 85, 336]` (对应 "5", "!", "v", "em")

1. **输入到目标模型**: `[prompt..., 20, 0, 85, 336]`
2. **目标模型生成 logits**:
   - `next_token_logits[0]`: 在 prompt 之后预测的 logits
   - `next_token_logits[1]`: 在 token 20 之后预测的 logits
   - `next_token_logits[2]`: 在 token 0 之后预测的 logits
   - `next_token_logits[3]`: 在 token 85 之后预测的 logits

3. **提取目标预测**:
   - `target_predict[0] = argmax(next_token_logits[0])` → 例如: 20 (验证 candidates[1])
   - `target_predict[1] = argmax(next_token_logits[1])` → 例如: 4 (验证 candidates[2])
   - `target_predict[2] = argmax(next_token_logits[2])` → 例如: 0 (验证 candidates[3])
   - `target_predict[3] = argmax(next_token_logits[3])` → 例如: 85

4. **验证比较**:
   - 位置0: `candidates[0]=20` vs `target_predict[?]` (需要特殊处理)
   - 位置1: `candidates[1]=0` vs `target_predict[0]=20` → 不匹配，拒绝
   - 位置2: `candidates[2]=85` vs `target_predict[1]=4` → 不匹配，拒绝
   - 位置3: `candidates[3]=336` vs `target_predict[2]=0` → 不匹配，拒绝

## 注意事项

1. **索引偏移**: `target_predict[i]` 验证的是 `candidates[i+1]`，不是 `candidates[i]`
2. **位置0特殊处理**: `candidates[0]` 的验证需要从 `predict` 数组或特殊逻辑获取
3. **接受是连续的**: 一旦某个位置被拒绝，后续位置都会被拒绝
4. **KV Cache**: 目标模型使用 KV cache 存储 prompt，只对草稿 tokens 进行前向传播

