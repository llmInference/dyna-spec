# 动态词汇表配置指南

## 概述

动态词汇表系统用于 draft model 的推测解码。系统从初始词汇表（由 `--init-vocab-size` 指定）开始，然后可以通过 API 动态添加或移除词汇。所有词汇表操作都是全局共享的，使用 LRU 策略管理容量。

## 1. 启动服务器（使用推测解码 + 动态词汇表）

### 使用 `sglang.launch_server` 启动

这个命令是将同一个模型同时作为基础模型和草稿模型，如果需要使用eagle3还需要修改--speculative-algorithm，但是好像会出现架构类型不存在的情况，可能之后还要修改一下。

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

**说明：**
- `--init-vocab-size 327`：指定动态词汇表的初始大小（从 token ID 0 开始的前 327 个 token）
- `--dyna-space 1024`：指定在初始词汇表基础上可以额外添加的 token 数量（默认：1024）

### 初始词汇表配置

动态词汇表从初始词汇表开始，通过 `--init-vocab-size` 参数指定初始大小。例如，如果词汇表总大小为 32768，想要使用前 327 个 token 作为初始词汇表，可以设置：

```bash
--init-vocab-size 327
```

**注意：** 
- `--init-vocab-size` 指定的是初始词汇表的 token 数量，从 token ID 0 开始到 `init_vocab_size - 1`
- 如果不指定 `--init-vocab-size`，将使用完整的词汇表大小作为初始词汇表
- `--dyna-space` 指定了在初始词汇表基础上可以额外添加的 token 数量，默认值为 1024

## 2. 启动 FastAPI 服务器（管理动态词汇表）

```bash
cd /home/llminference/syq/dyna-spec
export SGLANG_RUNTIME_URL=http://127.0.0.1:30001
python -m uvicorn sglang.api.api_server:app --host 0.0.0.0 --port 30000
```

## 3. 动态词汇表的使用

动态词汇表通过 API 端点管理，用于 draft model 的推理。**注意：** 动态词汇表是全局共享的，所有请求使用同一个词汇表。

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
    "prompt": "Hello, world",
    "max_new_tokens": 10,
    "temperature": 0.7
  }'
```

**重要：** 词汇表会在首次使用前自动初始化初始词汇表（从 token ID 0 到 `init_vocab_size - 1`）。之后可以通过 `/v1/vocab/add` 添加额外的词汇。

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

## 6. 动态词汇表系统特性

动态词汇表系统具有以下特性：

- **初始词汇表**：系统从初始词汇表开始（由 `--init-vocab-size` 指定，从 token ID 0 开始）
- **动态管理**：可以通过 API 在运行时添加或移除词汇
- **全局共享**：所有请求共享同一个动态词汇表
- **容量管理**：总容量 = 初始词汇表大小 + `dyna_space`，使用 LRU 策略自动管理
- **自动初始化**：在首次使用前自动初始化初始词汇表

## 7. 注意事项

1. **端口冲突**：确保 Runtime 服务器和 FastAPI 服务器使用不同端口
2. **全局共享**：动态词汇表是全局共享的，所有请求使用同一个词汇表。添加或移除词汇会影响所有后续请求
3. **自动初始化**：词汇表会在首次使用前自动初始化初始词汇表（从 token ID 0 到 `init_vocab_size - 1`）
4. **LRU 策略**：当词汇表超过容量（初始词汇表大小 + `dyna_space`）时，使用 LRU 策略自动移除最旧的 token
5. **Token ID 验证**：确保添加的词汇能够被 tokenizer 正确转换为有效的 token ID（必须在 `[0, vocab_size)` 范围内）
6. **初始词汇表**：初始词汇表是动态词汇表的起点，包含从 token ID 0 开始的前 `init_vocab_size` 个 token

## 8. 导出初始词汇表为 JSON

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

