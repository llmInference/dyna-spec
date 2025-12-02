# `accepted` 字段计算说明

## 概述

`accepted` 字段表示草稿模型在某个位置生成的 token 是否被目标模型接受。这个字段的值是通过推测解码的验证过程计算得出的。

## 计算流程

### 1. 验证过程

在 `eagle_info.py` 的 `verify` 方法中，验证过程如下：

```python
# 初始化 accept_index，形状为 (batch_size, spec_steps + 1)
# 初始值全部为 -1，表示未被接受
accept_index = torch.full(
    (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
)
```

### 2. 验证函数

根据采样策略（greedy 或 non-greedy），调用不同的验证函数：

#### Greedy 模式：
```python
predict, accept_index, accept_length = verify_tree_greedy_func(
    predicts=predict,
    accept_index=accept_index,  # 会被修改，填充被接受的索引
    accept_token_num=accept_length,
    candidates=candidates,  # 草稿模型生成的 tokens
    target_predict=target_predict,  # 目标模型预测的 tokens
    ...
)
```

#### Non-Greedy 模式：
```python
tree_speculative_sampling_target_only(
    predicts=predict,
    accept_index=accept_index,  # 会被修改，填充被接受的索引
    accept_token_num=accept_length,
    ...
)
```

### 3. `accept_index` 的含义

`accept_index` 是一个二维数组：
- 形状：`(batch_size, spec_steps + 1)`
- 每个元素 `accept_index[i, j]` 存储的是在 `predict` 数组中的**索引**
- 如果值为 `-1`，表示该位置未被接受
- 如果值不为 `-1`，表示该位置被接受，值指向 `predict` 数组中的位置

### 4. 从 `accept_index` 提取接受位置

```python
# 获取当前请求的 accept_index 行
accept_index_row = accept_index_cpu[i]  # 例如: [0, 2, -1, -1]

# 提取所有被接受的位置
accepted_positions = set()
for j, idx in enumerate(accept_index_row):
    if idx == -1:
        break  # 遇到 -1 表示后续都未被接受
    # idx 指向 predict 数组中的位置
    # 通过取模运算计算实际在 draft_token_num 中的位置
    pos = idx % self.draft_token_num
    accepted_positions.add(pos)
```

**示例：**
- 假设 `draft_token_num = 4`（每轮生成4个token）
- `accept_index_row = [0, 5, -1, -1]`
  - `idx=0`: `pos = 0 % 4 = 0` → 位置0被接受
  - `idx=5`: `pos = 5 % 4 = 1` → 位置1被接受
  - `idx=-1`: 停止，后续位置未被接受

### 5. 计算 `accepted` 字段

```python
# 对于每个位置，检查是否在 accepted_positions 中
for pos in range(min(len(req_candidates), len(req_target_predict))):
    draft_token_id = int(req_candidates[pos])
    target_token_id = int(req_target_predict[pos])
    
    # 判断该位置是否被接受
    is_accepted = pos in accepted_positions
    
    round_detail["verification_results"].append({
        "position": pos,
        "draft_token_id": draft_token_id,
        "target_token_id": target_token_id,
        "accepted": is_accepted  # 这里就是 accepted 字段的值
    })
```

## 验证逻辑

### Greedy 模式下的验证

在 greedy 模式下，验证逻辑是：
1. 目标模型对每个位置计算 logits
2. 取 argmax 得到目标模型预测的 token
3. 比较草稿模型的 token 和目标模型的 token：
   - 如果相同 → 接受
   - 如果不同 → 拒绝

### Non-Greedy 模式下的验证

在 non-greedy 模式下，使用 rejection sampling：
1. 计算目标模型和草稿模型的概率分布
2. 使用阈值（`threshold_single`, `threshold_acc`）进行采样
3. 根据采样结果决定是否接受

## 关键点

1. **`accept_index` 存储的是索引，不是位置**
   - 需要通过 `idx % draft_token_num` 转换为实际位置

2. **`-1` 表示未被接受**
   - 当遇到 `-1` 时，后续位置都未被接受

3. **接受是连续的**
   - 如果位置 i 被接受，那么位置 0 到 i-1 也都被接受
   - 这是推测解码的特性：一旦某个位置被拒绝，后续位置都会被拒绝

4. **`accepted` 字段是布尔值**
   - `True`: 该位置的草稿 token 被目标模型接受
   - `False`: 该位置的草稿 token 被目标模型拒绝

## 示例

假设一轮验证中有4个位置：

```python
# 草稿模型生成的 tokens
candidates = [20, 0, 85, 336]  # ["5", "!", "v", "em"]

# 目标模型预测的 tokens
target_predict = [20, 4, 0, 0]  # ["5", "%", "!", "!"]

# accept_index_row = [0, -1, -1, -1]
# 表示只有位置0被接受

# 结果：
verification_results = [
    {"position": 0, "draft_token_id": 20, "target_token_id": 20, "accepted": True},   # 匹配，接受
    {"position": 1, "draft_token_id": 0, "target_token_id": 4, "accepted": False},  # 不匹配，拒绝
    {"position": 2, "draft_token_id": 85, "target_token_id": 0, "accepted": False},  # 不匹配，拒绝
    {"position": 3, "draft_token_id": 336, "target_token_id": 0, "accepted": False}, # 不匹配，拒绝
]
```

## 相关代码位置

- `eagle_info.py:406-415`: 从 `accept_index` 提取接受位置
- `eagle_info.py:417-431`: 计算每个位置的 `accepted` 值
- `eagle_utils.py:345-383`: 验证函数的实现

