# 动态词汇表配置指南

## 概述

动态词汇表系统用于 draft model 的推测解码。系统从初始词汇表（由 `--init-vocab-size` 指定）开始，然后可以通过 API 动态添加或移除词汇。所有词汇表操作都是全局共享的，使用 LRU 策略管理容量。

**重要特性：**
- **动态词汇表仅用于草稿模型**：动态词汇表只会应用到草稿模型（draft model），目标模型（target model）始终使用完整的词汇表，以确保输出结果的合理性和质量。
- **支持自定义词汇表**：可以传入同模型 tokenizer 在不同数据集下分词得到的词汇表，该词汇表也仅用于草稿模型。

## 1. 启动服务器（使用推测解码 + 动态词汇表）

### 使用 `sglang.launch_server` 启动

这个命令是将同一个模型同时作为基础模型和草稿模型，如果需要使用eagle3还需要修改--speculative-algorithm，但是好像会出现架构类型不存在的情况，可能之后还要修改一下。

#### 模式一：使用 `--init-vocab-size`（指定初始词汇表大小）

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen3-1.7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --max-total-tokens 16384 \
    --init-vocab-size 327 \
    --dyna-space 1024 \
    --enable-return-hidden-states \
    --port 30001
```

**说明：**
- `--init-vocab-size 327`：指定动态词汇表的初始大小（从 token ID 0 开始的前 327 个 token）
- `--dyna-space 1024`：指定在初始词汇表基础上可以额外添加的 token 数量（默认：1024）
- `--enable-return-hidden-states`:可以实现拉取最后隐藏层特征向量

#### 模式二：使用 `--custom-vocab`（从文件加载自定义词汇表）

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen3-1.7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --max-total-tokens 16384 \
    --custom-vocab ./client/vocab_3000.json \
    --dyna-space 1024 \
    --enable-return-hidden-states \
    --port 30001
```

autodl 4090
```
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm STANDALONE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --max-total-tokens 8192 \
  --mem-fraction-static 0.5 \
  --cuda-graph-max-bs 8 \
  --custom-vocab ./client/vocab_3000.json \
  --dyna-space 1024 \
  --enable-return-hidden-states \
  --port 30001
  ```

**说明：**
- `--custom-vocab /path/to/custom_vocab.json`：指定自定义词汇表文件路径（JSON格式）
- `--dyna-space 1024`：指定在自定义词汇表基础上可以额外添加的 token 数量（默认：1024）
- `--enable-return-hidden-states`:可以实现拉取最后隐藏层特征向量

**自定义词汇表文件格式：**

支持两种JSON格式：

1. **对象格式**（推荐）：
```json
{
  "token_ids": [0, 1, 2, 100, 200, 300, 500, 1000]
}
```

2. **数组格式**：
```json
[0, 1, 2, 100, 200, 300, 500, 1000]
```

**注意：** 
- `--init-vocab-size` 和 `--custom-vocab` 是互斥的，不能同时使用
- 如果不指定 `--init-vocab-size` 也不指定 `--custom-vocab`，将使用完整的词汇表大小作为初始词汇表
- `--dyna-space` 指定了在初始词汇表基础上可以额外添加的 token 数量，默认值为 1024
- 自定义词汇表仅用于草稿模型，目标模型始终使用完整词汇表

## 2. 启动 FastAPI 服务器（管理动态词汇表）

```bash
cd /home/llminference/syq/dyna-spec
export SGLANG_RUNTIME_URL=http://127.0.0.1:30001
python -m uvicorn sglang.api.api_server:app --host 0.0.0.0 --port 30000
```

## 3. 动态词汇表的使用

动态词汇表通过 API 端点管理，用于 draft model 的推理。**注意：** 动态词汇表是全局共享的，所有请求使用同一个词汇表。

**重要：动态词汇表仅用于草稿模型**
- 动态词汇表只会应用到草稿模型（draft model）的推理过程
- 目标模型（target model）始终使用完整的词汇表，以确保输出结果的合理性和质量
- 这意味着即使草稿模型使用受限的词汇表，目标模型的验证和最终输出仍然基于完整词汇表

### 查询当前词汇表状态

```bash
curl -X GET http://127.0.0.1:30000/v1/vocab/query
```

返回示例：
```json
{
  "status": "success",
  "vocab_count": 328,
  "initial_vocab_count": 327
}
```

- `vocab_count`：当前动态词汇表中的 token 数量（包括初始词汇表和后续添加的词汇）
- `initial_vocab_count`：初始词汇表的 token 数量（由 `--init-vocab-size` 指定）

### 添加词汇到动态词汇表

```bash
curl -X POST http://127.0.0.1:30000/v1/vocab/add \
  -H "Content-Type: application/json" \
  -d '{
    "words": ["thing", "pty", "Key"]
  }'
```

### 从动态词汇表移除词汇

```bash
curl -X POST http://127.0.0.1:30000/v1/vocab/remove \
  -H "Content-Type: application/json" \
  -d '{
    "words": ["ismo"]
  }'
```

### 使用动态词汇表进行生成

```bash
curl -X POST http://127.0.0.1:30000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "1,2,3,4,",
    "max_new_tokens": 10,
    "temperature": 0.7
  }'
```

示例：可以返回模型输出的内容和草稿模型的平均接受长度
```
(sglang2) root@autodl-container-0bc2448e04-fa8499c0:~/autodl-tmp/syq/d
yna-spec# curl -X POST http://127.0.0.1:30000/v1/generate   -H "Content-Type: ap
plication/json"   -d '{
    "prompt": "what is the capital of france?",
    "max_new_tokens": 10,
    "temperature": 0.7
  }'
{"text":" (answer with just the name of the city,","draft_avg_accept_length":1.6666666666666667}
```

**重要：** 词汇表会在首次使用前自动初始化初始词汇表（从 token ID 0 到 `init_vocab_size - 1`）。之后可以通过 `/v1/vocab/add` 添加额外的词汇。

### 获取隐藏层特征（可选）

如果需要拉取 Transformer 最后一层的 hidden states（以及完整 meta 信息），可以使用新增的 `/v1/hidden_states`：

```bash
curl -X POST http://127.0.0.1:30000/v1/hidden_states \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Need transformer features"
  }'
```

- 与 `/v1/generate` 一样会自动复用动态词汇表；
- 若未显式指定 `max_new_tokens`，默认只做前向（`max_new_tokens=0`）；
- 返回 JSON：
  ```json
  {
    "hidden_states": [[...], [...]],
    "meta_info": {
      "hidden_states": [[...], [...]],
      "prompt_tokens": 15,
      "completion_tokens": 0,
      ...
    }
  }
  ```

> ⚠️ 运行时需以 `--enable-return-hidden-states` 启动，否则该接口会提示未开启。

## 4. 动态词汇表的范围限制

### 默认容量

全局动态词汇表的默认容量为 **初始词汇表大小 + dyna_space 个 token**。也就是说，在加载初始词汇表后，仍然可以额外添加 `dyna_space` 个新 token（默认 1024）。当超过容量时，会使用 LRU（最近最少使用）策略自动移除最旧的 token。

### 修改容量

可以通过 `--dyna-space` 参数修改动态空间大小：

```bash
--dyna-space 2048  # 允许额外 2048 个 token
```

### Token ID 范围验证

动态词汇表中的 token ID 必须在基础词汇表的有效范围内（通常是 `[0, vocab_size)`）。API 服务器会通过 tokenizer 将单词转换为 token ID，确保 ID 的有效性。

## 5. 自定义词汇表与目标模型词汇表的映射机制

### 映射原理

当使用自定义词汇表时，系统会自动处理草稿模型和目标模型之间的 token ID 映射：

1. **映射表结构**：
   - `dynamic_vocab_token_ids` 是一个一维数组，存储了从局部索引到全局 token ID 的映射
   - 数组索引 = 草稿模型的局部索引（0, 1, 2, ..., len(custom_vocab)-1）
   - 数组值 = 目标模型的全局 token ID（来自自定义词汇表）

2. **映射过程**：
   ```
   草稿模型推理 → 局部索引 (0, 1, 2, ...)
                    ↓
   映射函数 _map_dynamic_vocab_token_ids()
                    ↓
   全局 token ID (来自 dynamic_vocab_token_ids)
                    ↓
   目标模型验证 → 使用全局 token ID
   ```

3. **代码实现**：
   ```python
   # 在 sampler.py 中的映射函数
   def _map_dynamic_vocab_token_ids(
       self,
       batch_next_token_ids: torch.Tensor,  # 局部索引
       dynamic_vocab_token_ids: torch.Tensor,  # 映射表
   ) -> torch.Tensor:
       # 通过索引查找全局 token ID
       return dynamic_vocab_token_ids[batch_next_token_ids]
   ```

### 实际示例

假设你有一个自定义词汇表文件 `custom_vocab.json`：
```json
{
  "token_ids": [0, 100, 200, 300, 500, 1000]
}
```

映射关系如下：

| 局部索引 | 全局 token ID | 说明 |
|---------|--------------|------|
| 0 | 0 | 草稿模型采样索引0 → 目标模型token ID 0 |
| 1 | 100 | 草稿模型采样索引1 → 目标模型token ID 100 |
| 2 | 200 | 草稿模型采样索引2 → 目标模型token ID 200 |
| 3 | 300 | 草稿模型采样索引3 → 目标模型token ID 300 |
| 4 | 500 | 草稿模型采样索引4 → 目标模型token ID 500 |
| 5 | 1000 | 草稿模型采样索引5 → 目标模型token ID 1000 |

**工作流程**：
1. 草稿模型在自定义词汇表中生成 token，假设采样得到局部索引 `2`
2. 系统自动查找 `dynamic_vocab_token_ids[2] = 200`
3. 将全局 token ID `200` 传递给目标模型进行验证
4. 目标模型使用完整的词汇表验证 token ID `200` 的概率

### 验证机制

系统在加载自定义词汇表时会自动验证：

1. **范围检查**：所有 token ID 必须在 `[0, vocab_size)` 范围内
2. **重复检查**：自动去除重复的 token ID
3. **无效值处理**：超出范围的 token ID 会被忽略并记录警告

**示例日志**：
```
Loaded 500 static vocab tokens from JSON file custom_vocab.json (3 invalid entries omitted).
```

### 重要提示

1. **Token ID 必须是有效的**：
   - 自定义词汇表中的所有 token ID 必须是目标模型词汇表中的有效 token
   - 系统会在加载时验证，但建议在准备词汇表时就确保正确性

2. **不需要连续**：
   - Token ID 不需要连续，可以是任意顺序
   - 例如：`[0, 100, 200, 300]` 是完全有效的

3. **自动映射**：
   - 映射过程完全自动，无需手动配置
   - 系统保证草稿模型生成的 token 能正确映射到目标模型

4. **同模型 tokenizer**：
   - 建议使用与目标模型相同的 tokenizer 来生成自定义词汇表
   - 这样可以确保所有 token ID 都是有效的

## 5. 完整示例

### 步骤 1：启动 SGLang Runtime 服务器

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen3-1.7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --max-total-tokens 16384 \
    --init-vocab-size 327 \
    --dyna-space 1024 \
    --port 30001
```

### 步骤 2：启动 FastAPI 服务器

```bash
export SGLANG_RUNTIME_URL=http://127.0.0.1:30001
python -m uvicorn sglang.api.api_server:app --host 0.0.0.0 --port 30000
```

### 步骤 3：使用 Python 客户端

```python
import requests

BASE_URL = "http://127.0.0.1:30000"

# 查询当前词汇表状态
response = requests.get(f"{BASE_URL}/v1/vocab/query")
print(f"Vocab status: {response.json()}")

# 添加额外词汇到动态词汇表（初始词汇表已自动初始化）
vocab_words = ["token-alpha", "token-beta", "token-gamma"]
response = requests.post(
    f"{BASE_URL}/v1/vocab/add",
    json={"words": vocab_words}
)
print(f"Added vocab: {response.json()}")

# 使用动态词汇表生成
response = requests.post(
    f"{BASE_URL}/v1/generate",
    json={
        "prompt": "Testing dynamic vocab",
        "max_new_tokens": 10,
        "temperature": 0.7
    }
)
print(f"Generated: {response.json()}")
```

## 7. 动态词汇表系统特性

动态词汇表系统具有以下特性：

- **初始词汇表**：系统从初始词汇表开始（由 `--init-vocab-size` 指定，从 token ID 0 开始）
- **动态管理**：可以通过 API 在运行时添加或移除词汇
- **全局共享**：所有请求共享同一个动态词汇表
- **容量管理**：总容量 = 初始词汇表大小 + `dyna_space`，使用 LRU 策略自动管理
- **自动初始化**：在首次使用前自动初始化初始词汇表
- **仅用于草稿模型**：动态词汇表只应用于草稿模型，目标模型始终使用完整词汇表
- **支持自定义词汇表**：可以加载同模型 tokenizer 在不同数据集下分词得到的词汇表

## 8. 注意事项

1. **端口冲突**：确保 Runtime 服务器和 FastAPI 服务器使用不同端口
2. **全局共享**：动态词汇表是全局共享的，所有请求使用同一个词汇表。添加或移除词汇会影响所有后续请求
3. **自动初始化**：词汇表会在首次使用前自动初始化初始词汇表（从 token ID 0 到 `init_vocab_size - 1`）
4. **LRU 策略**：当词汇表超过容量（初始词汇表大小 + `dyna_space`）时，使用 LRU 策略自动移除最旧的 token
5. **Token ID 验证**：确保添加的词汇能够被 tokenizer 正确转换为有效的 token ID（必须在 `[0, vocab_size)` 范围内）
6. **初始词汇表**：初始词汇表是动态词汇表的起点，包含从 token ID 0 开始的前 `init_vocab_size` 个 token

## 9. 导出初始词汇表为 JSON

如果你需要在调试或配置时查看初始词汇表的前若干 token，可以使用 `client/dump_static_vocab.py` 脚本快速导出：

```bash
cd /home/llminference/syq/dyna-spec
python client/dump_static_vocab.py \
  --count 1600 \
  --output client/vocab_top1600.json \
  --runtime-url http://127.0.0.1:30001
```

- `--count`：导出的 token 数量，默认为 1600
- `--output`：输出 JSON 文件路径，默认为 `client/vocab_top1600.json`
- `--runtime-url`：可选，指定正在运行的 SGLang Runtime（如果已经通过 `sglang.set_default_backend` 连接，可省略）

执行后会生成如下结构的 JSON 文件：

```json
{
  "count": 1600,
  "runtime_url": "http://127.0.0.1:30001",
  "tokens": [
    {"token_id": 0, "token": "<|endoftext|>"},
    {"token_id": 1, "token": "the"},
    ...
  ]
}
```

可以将该文件用于查看和验证初始词汇表的配置。

