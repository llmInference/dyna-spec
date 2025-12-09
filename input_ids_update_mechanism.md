# EAGLE 中 input_ids 的更新机制

本文档总结了在 EAGLE 推测解码中 `input_ids` 的更新机制。

## 更新位置概览

`input_ids` 在以下四个关键位置被更新：

1. **验证阶段** (`EagleVerifyInput.prepare_for_verify`)
2. **扩展阶段** (`EagleDraftInput.prepare_for_extend`)
3. **解码后扩展阶段** (`EagleDraftInput.prepare_extend_after_decode`)
4. **Draft Forward 阶段** (`EAGLEWorker.draft_forward`)

---

## 1. 验证阶段：`EagleVerifyInput.prepare_for_verify`

**文件**: `eagle_info.py:106-150`

**更新机制**:
```python
batch.input_ids = self.draft_token
```

**说明**:
- 在验证阶段，将 draft model 生成的候选 token (`draft_token`) 设置为 batch 的 `input_ids`
- 这些 draft token 将被 target model 验证
- 发生在 `EAGLEWorker.verify()` 方法中，调用 `spec_info.prepare_for_verify(batch, self.page_size)`

**调用链**:
```
EAGLEWorker.verify() 
  -> spec_info.prepare_for_verify(batch, page_size)
    -> batch.input_ids = self.draft_token
```

---

## 2. 扩展阶段：`EagleDraftInput.prepare_for_extend`

**文件**: `eagle_info.py:1025-1039`

**更新机制**:
```python
pt = 0
for i, extend_len in enumerate(batch.extend_lens):
    input_ids = batch.input_ids[pt : pt + extend_len]
    batch.input_ids[pt : pt + extend_len] = torch.cat(
        (input_ids[1:], self.verified_id[i].reshape(1))
    )
    pt += extend_len
```

**说明**:
- 滑动窗口更新：移除每个序列的第一个 token，在末尾添加新验证的 token (`verified_id`)
- 用于 draft model 的 extend 阶段，准备下一次 draft 生成的输入
- 发生在 `EAGLEWorker.forward_draft_extend()` 中

**示例**:
- 原始: `[token1, token2, token3]`
- 更新后: `[token2, token3, verified_token]`

**调用链**:
```
EAGLEWorker.forward_draft_extend()
  -> batch.spec_info.prepare_for_extend(batch)
    -> batch.input_ids[pt:pt+extend_len] = torch.cat((input_ids[1:], verified_id))
```

---

## 3. 解码后扩展阶段：`EagleDraftInput.prepare_extend_after_decode`

**文件**: `eagle_info.py:1060-1090`

**更新机制**:
```python
batch.input_ids = self.verified_id
```

**说明**:
- 直接将验证通过的 token (`verified_id`) 设置为 `input_ids`
- 用于在 decode 阶段后，为 draft model 的 extend 操作准备输入
- 发生在 `EAGLEWorker.forward_draft_extend_after_decode()` 中

**调用链**:
```
EAGLEWorker.forward_draft_extend_after_decode()
  -> batch.spec_info.prepare_extend_after_decode(batch, speculative_num_steps)
    -> batch.input_ids = self.verified_id
```

---

## 4. Draft Forward 阶段：`EAGLEWorker.draft_forward`

**文件**: `eagle_worker.py:573-643`

**更新机制**:
```python
for i in range(self.speculative_num_steps):
    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
        i, topk_p, topk_index, hidden_states, scores, self.topk
    )
    # ...
    forward_batch.input_ids = input_ids
    # ...
    # Run forward
    logits_output, _ = self.draft_model_runner.forward(forward_batch, ...)
```

**说明**:
- 在 draft model 的多步前向传播中，每次迭代更新 `forward_batch.input_ids`
- `input_ids` 通过 `select_top_k_tokens()` 从 topk 候选中选择
- 用于 draft model 生成候选 token 树

**调用链**:
```
EAGLEWorker.draft()
  -> EAGLEWorker.draft_forward(forward_batch)
    -> for i in range(speculative_num_steps):
         forward_batch.input_ids = input_ids  # 每次迭代更新
```

---

## 完整流程示例

### 典型的 EAGLE 解码流程：

1. **Draft 阶段** (`EAGLEWorker.draft()`):
   - 使用当前的 `batch.input_ids` 和 hidden states
   - 生成 draft token 树
   - 在 `draft_forward()` 中更新 `forward_batch.input_ids`

2. **Verify 阶段** (`EAGLEWorker.verify()`):
   - `prepare_for_verify()`: `batch.input_ids = draft_token`
   - Target model 验证 draft tokens
   - 生成 `verified_id` (接受的 tokens)

3. **Extend After Decode** (`EAGLEWorker.forward_draft_extend_after_decode()`):
   - `prepare_extend_after_decode()`: `batch.input_ids = verified_id`
   - 为下一次 draft 准备输入

4. **Extend 阶段** (`EAGLEWorker.forward_draft_extend()`):
   - `prepare_for_extend()`: 滑动窗口更新 `batch.input_ids`
   - 移除第一个 token，添加新的 verified token

---

## 关键数据结构

- `batch.input_ids`: 当前 batch 的输入 token IDs
- `self.draft_token`: Draft model 生成的候选 token IDs
- `self.verified_id`: Target model 验证通过的 token IDs
- `forward_batch.input_ids`: Draft forward 过程中的输入 token IDs

---

## 注意事项

1. **验证阶段**: `batch.input_ids` 被设置为所有 draft tokens，用于 target model 验证
2. **扩展阶段**: 使用滑动窗口机制，保持序列长度不变
3. **解码后扩展**: 直接使用验证通过的 tokens 作为下一次 draft 的输入
4. **Draft Forward**: 在每次迭代中动态更新，从 topk 候选中选择

---

## 相关代码位置

- `eagle_info.py:111` - `prepare_for_verify` 中的更新
- `eagle_info.py:1035-1038` - `prepare_for_extend` 中的滑动窗口更新
- `eagle_info.py:1069` - `prepare_extend_after_decode` 中的更新
- `eagle_worker.py:613` - `draft_forward` 中的迭代更新

