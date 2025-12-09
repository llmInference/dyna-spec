# verified_id 的生成和添加机制

本文档详细说明在 EAGLE 验证阶段，`verified_id`（验证通过的 token IDs）是如何被生成和添加的。

## 核心流程概览

`verified_id` 的生成和添加发生在 `EagleVerifyInput.verify()` 方法中，主要包含以下步骤：

1. **生成 predict 和 accept_index** - 通过验证算法确定哪些 draft tokens 被接受
2. **添加到 req.output_ids** - 将接受的 tokens 添加到每个请求的输出序列中
3. **提取 verified_id** - 从 predict 中提取所有接受的 token IDs
4. **返回给调用者** - 通过 `EagleVerifyOutput` 返回 verified_id

---

## 详细代码流程

### 1. 验证算法生成 predict 和 accept_index

**位置**: `eagle_info.py:626-705`

验证过程通过两种方式之一进行：

#### A. Greedy 验证（贪婪模式）

```python
# 第636-648行
target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
target_predict = target_predict.reshape(bs, self.draft_token_num)
predict, accept_index, accept_length = verify_tree_greedy_func(
    predicts=predict,  # mutable - 输出：target model 预测的 tokens
    accept_index=accept_index,  # mutable - 输出：接受的 token 在 predict 中的索引
    accept_token_num=accept_length,  # mutable - 输出：每个请求接受的 token 数量
    candidates=candidates,  # draft model 生成的候选 tokens
    retrive_index=self.retrive_index,
    retrive_next_token=self.retrive_next_token,
    retrive_next_sibling=self.retrive_next_sibling,
    target_predict=target_predict,  # target model 的预测
    topk=self.topk,
)
```

#### B. 采样验证（非贪婪模式）

```python
# 第690-705行
tree_speculative_sampling_target_only(
    predicts=predict,  # mutable
    accept_index=accept_index,  # mutable
    accept_token_num=accept_length,  # mutable
    candidates=candidates,
    retrive_index=self.retrive_index,
    retrive_next_token=self.retrive_next_token,
    retrive_next_sibling=self.retrive_next_sibling,
    uniform_samples=coins,
    uniform_samples_for_final_sampling=coins_for_final_sampling,
    target_probs=target_probs,
    draft_probs=draft_probs,
    threshold_single=...,
    threshold_acc=...,
    deterministic=True,
)
```

**关键数据结构**:
- `predict`: shape `(total_tokens,)` - 包含 target model 预测的所有 token IDs（包括接受的）
- `accept_index`: shape `(bs, spec_steps + 1)` - 每个请求接受的 token 在 predict 中的索引，-1 表示未接受
- `accept_length`: shape `(bs,)` - 每个请求接受的 token 数量

---

### 2. 将接受的 tokens 添加到 req.output_ids

**位置**: `eagle_info.py:728-758`

这是 **第一个添加位置**，直接将接受的 tokens 添加到每个请求的输出序列中：

```python
# 第728-733行
for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
    for j, idx in enumerate(accept_index_row):
        if idx == -1:
            break
        id = predict_cpu[idx]  # 从 predict 中获取 token ID
        req.output_ids.append(id)  # ⭐ 添加到请求的输出序列
        req.check_finished()
        if req.finished():
            has_finished = True
            # set all tokens after finished token to -1 and break
            accept_index[i, j + 1 :] = -1
            break
        else:
            if req.grammar is not None:
                try:
                    req.grammar.accept_token(id)
                except ValueError as e:
                    # ...
                    raise e
```

**说明**:
- 遍历每个请求的 `accept_index_row`（接受的 token 索引）
- 从 `predict` 中提取对应的 token ID
- **直接添加到 `req.output_ids`** - 这是 tokens 被添加到输出序列的地方
- 检查请求是否完成，如果完成则停止添加后续 tokens

---

### 3. 提取 verified_id

**位置**: `eagle_info.py:792-797`

在将所有接受的 tokens 添加到 `req.output_ids` 后，从 `predict` 中提取所有接受的 token IDs 作为 `verified_id`：

```python
# 第794-797行
accept_index = accept_index[accept_index != -1]  # 移除 -1，只保留有效索引
verified_id = predict[accept_index]  # ⭐ 从 predict 中提取所有接受的 tokens
logger.info(f"accept_index: {accept_index}")
logger.info(f"verified_id: {verified_id}")
```

**说明**:
- `accept_index` 被展平并过滤掉 -1（未接受的标记）
- `verified_id = predict[accept_index]` 提取所有接受的 token IDs
- `verified_id` 是一个 1D tensor，包含所有接受的 tokens（跨所有请求）

---

### 4. 返回 verified_id

**位置**: `eagle_info.py:887-903` 和 `970-976`

`verified_id` 被包含在 `EagleVerifyOutput` 中返回：

```python
# 第897-903行（未完成的请求）
return EagleVerifyOutput(
    draft_input=draft_input,
    logits_output=logits_output,
    verified_id=verified_id,  # ⭐ 返回验证通过的 tokens
    accept_length_per_req_cpu=draft_input.accept_length_cpu,
    accepted_indices=accept_index,
)

# 第970-976行（有完成的请求）
return EagleVerifyOutput(
    draft_input=draft_input,
    logits_output=logits_output,
    verified_id=verified_id,  # ⭐ 返回验证通过的 tokens
    accept_length_per_req_cpu=accept_length_list,
    accepted_indices=accept_index,
)
```

同时，`verified_id` 也被传递给 `EagleDraftInput`：

```python
# 第887-895行
draft_input = EagleDraftInput(
    hidden_states=batch.spec_info.hidden_states[accept_index],
    verified_id=verified_id,  # ⭐ 传递给下一次 draft 的输入
    accept_length=accept_length,
    accept_length_cpu=accept_length_list,
    seq_lens_for_draft_extend=batch.seq_lens,
    seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu,
    req_pool_indices_for_draft_extend=batch.req_pool_indices,
)
```

---

## 完整调用链

```
EAGLEWorker.verify()
  └─> EagleVerifyInput.verify()
      ├─> verify_tree_greedy_func() 或 tree_speculative_sampling_target_only()
      │   └─> 生成 predict 和 accept_index
      │
      ├─> for req in batch.reqs:
      │   └─> req.output_ids.append(id)  ⭐ 第一个添加位置
      │
      ├─> verified_id = predict[accept_index]  ⭐ 提取 verified_id
      │
      └─> return EagleVerifyOutput(verified_id=verified_id)
          └─> draft_input = EagleDraftInput(verified_id=verified_id)
```

---

## 关键点总结

### verified_id 的生成位置

1. **生成阶段**: `eagle_info.py:638-648` 或 `690-705`
   - 通过 `verify_tree_greedy_func()` 或 `tree_speculative_sampling_target_only()` 生成 `predict` 和 `accept_index`

2. **提取阶段**: `eagle_info.py:795`
   - `verified_id = predict[accept_index]`
   - 从 `predict` 中根据 `accept_index` 提取所有接受的 tokens

### verified_id 的添加位置

1. **添加到 req.output_ids**: `eagle_info.py:733`
   - `req.output_ids.append(id)` - 这是 tokens 被添加到输出序列的**主要位置**
   - 发生在验证循环中，逐个添加每个接受的 token

2. **传递给 EagleDraftInput**: `eagle_info.py:889` 或 `952`
   - `EagleDraftInput(verified_id=verified_id)` - 用于下一次 draft 生成

3. **返回给调用者**: `eagle_info.py:900` 或 `973`
   - `EagleVerifyOutput(verified_id=verified_id)` - 返回给 `EAGLEWorker.verify()`

---

## 数据流示例

假设有 2 个请求，每个请求接受了 3 个 tokens：

1. **验证后**:
   - `predict = [token1, token2, token3, token4, token5, token6]` (6 个 tokens)
   - `accept_index = [[0, 1, 2], [3, 4, 5]]` (每个请求接受 3 个)
   - `accept_index` (展平后) = `[0, 1, 2, 3, 4, 5]`

2. **添加到 req.output_ids**:
   - `req[0].output_ids.append(token1)`
   - `req[0].output_ids.append(token2)`
   - `req[0].output_ids.append(token3)`
   - `req[1].output_ids.append(token4)`
   - `req[1].output_ids.append(token5)`
   - `req[1].output_ids.append(token6)`

3. **提取 verified_id**:
   - `verified_id = predict[[0, 1, 2, 3, 4, 5]]`
   - `verified_id = [token1, token2, token3, token4, token5, token6]`

4. **返回和使用**:
   - `EagleVerifyOutput.verified_id = verified_id`
   - `EagleDraftInput.verified_id = verified_id` (用于下一次 draft)

---

## 相关代码位置

- **生成 predict**: `eagle_info.py:584` - 初始化 predict tensor
- **验证算法**: `eagle_info.py:638-648` (greedy) 或 `690-705` (sampling)
- **添加到 output_ids**: `eagle_info.py:733` - `req.output_ids.append(id)`
- **提取 verified_id**: `eagle_info.py:795` - `verified_id = predict[accept_index]`
- **返回 verified_id**: `eagle_info.py:889, 900, 952, 973`

