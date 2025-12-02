### **SGlang 词汇表优化功能开发指南**

#### **1. 项目背景与目标**

SGlang 作为一个高性能的推理框架，支持推测解码（Speculative Decoding）以加速 LLM 的推理过程。在推测解码中，一个轻量级的草稿模型（Draft Model）会并行地生成多个候选词元（Tokens），然后由主模型（Target Model）一次性进行验证。这种方法的效率在很大程度上取决于草稿模型的准确性和速度。

本项目旨在通过优化草稿模型在前向传播中处理的词汇表（Vocabulary）大小，来提升其推理速度和效率。一个完整的词汇表通常包含数万甚至数十万个词元，但在特定场景或语言中，高频使用的词元占比较小。通过将计算（如 LM Head 和 Softmax）限制在一个更小的、经过优化的词汇表子集上，我们可以显著减少计算开销和内存占用，从而加速草稿模型的生成过程。

我们将分三个阶段来实现一个从简单到复杂的、高度可配置的词汇表优化框架：

*   **第一阶段：** 实现一个基于比例的静态词汇表。
*   **第二阶段：** 扩展支持用户自定义的静态词汇表。
*   **第三阶段：** 构建一个“静态基础+动态扩展”的混合词汇表管理框架。

#### **2. 文件结构与代码定位**

为了实现此功能，我们需要对 SGlang 的核心代码进行修改。以下是本次开发涉及的关键文件和模块的路径（基于典型的 SGlang 项目结构）：

*   **模型配置**：草稿模型的配置文件通常位于 `python/sglang/srt/model_config.py` 或类似的配置管理模块中。我们需要在这里添加新的配置参数。
*   **模型定义与前向传播**：草稿模型（例如 Llama）的实现位于 `python/sglang/srt/models/llama.py`。核心的修改将集中在该文件的 `LlamaForCausalLM` 类的 `forward` 方法中，特别是 Logits 计算部分。
*   **模型初始化**：模型的加载和初始化逻辑通常在 `python/sglang/srt/models/load_model.py` 或模型类自身的构造函数中。我们需要在这里加入加载自定义词汇表的逻辑。

#### **3. 开发阶段详解**

##### **第一阶段：实现基于比例的静态词汇表**

此阶段的目标是快速实现一个基础功能，通过配置参数启用一个静态的、按比例缩小的词汇表。

**3.1. 修改模型配置**

在 `python/sglang/srt/model_config.py`（或相关配置文件）中，为模型配置类（如 `ModelConfig`）添加两个新参数：

```python
# python/sglang/srt/model_config.py

class ModelConfig:
    # ... 现有参数 ...
    vocab_size: int

    # --- 新增参数 ---
    # 是否为草稿模型启用静态词汇表优化
    use_static_vocab: bool = False
    # 静态词汇表的比例 (0.0, 1.0]，默认为 0.5
    static_vocab_ratio: float = 0.5
    # (将在第二阶段使用)
    custom_vocab_path: Optional[str] = None
```

**3.2. 修改草稿模型前向传播逻辑**

在草稿模型的 `forward` 方法中，我们需要根据配置决定是否启用词汇表裁剪。

**伪代码/代码片段** (以 `python/sglang/srt/models/llama.py` 中的 `LlamaForCausalLM.forward` 为例):

```python
# python/sglang/srt/models/llama.py

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # ... 其他初始化 ...
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # --- 新增：初始化静态词汇表索引 ---
        self.static_vocab_indices = None
        if config.use_static_vocab and 0.0 < config.static_vocab_ratio <= 1.0:
            static_vocab_size = int(config.vocab_size * config.static_vocab_ratio)
            # 创建一个包含前 N% 词元 ID 的张量
            self.static_vocab_indices = torch.arange(static_vocab_size, device="cuda")

    def forward(self, input_ids, position_ids, seq_lens, ...):
        # ... 模型主干网络的前向传播 ...
        hidden_states = self.model(...)

        # --- 修改 Logits 计算逻辑 ---
        if self.static_vocab_indices is not None:
            # 1. 从原始 lm_head 权重中仅选择静态词汇表对应的部分
            #    注意：为了效率，此操作应在模型初始化时完成一次，或使用高效的索引操作
            active_lm_head_weight = self.lm_head.weight.index_select(0, self.static_vocab_indices)

            # 2. 在裁剪后的词汇表子集上计算 Logits
            logits = torch.matmul(hidden_states, active_lm_head_weight.t())

            # 3. (重要) Softmax 和采样将在此缩减的 Logits 上进行。
            #    后续的采样步骤需要知道原始的 token_id，因此需要将采样出的索引映射回原始词汇表 ID。
            #    例如：sampled_index -> self.static_vocab_indices[sampled_index]
        else:
            # 原始逻辑：在完整词汇表上计算 Logits
            logits = self.lm_head(hidden_states)

        return logits, ...
```

**关键点**：
*   `static_vocab_indices` 在模型初始化时被计算并缓存，避免重复计算。
*   通过 `index_select` 或类似的高效索引方法来获取 `lm_head` 的权重子集，而不是在每次 `forward` 时都重新切片。
*   后续的采样步骤（如 `torch.multinomial`）会得到一个在子集范围内的索引，必须将其映射回全局词汇表中的真实 `token_id`。

---


##### **附加需求：基于 SlimPajama 语料按出现率生成静态词汇表（新增）**

为方便批量构建并复用高频词元集合，我们新增一项明确需求：提供一个独立的工具（或脚本），用于从 SlimPajama 语料库中按 token 出现频率，生成给定大小 k 的静态词汇表文件（模型 token id，按频率从高到低排序，每行一个整数）。该文件可直接作为 `--speculative-static-vocab-path` / `custom_vocab_path` 的输入。

合同（Contract）
- 输入：
    - SlimPajama 语料路径（文件或目录，支持多文件流式读取）。
    - 模型分词器（tokenizer）的路径或名称（用于将文本分词并获得 token id）。
    - 目标词表大小 k（正整数，表示输出前 k 个最频繁的 token id）。
    - 输出路径（plain text 文件），每行包含一个 token id。
    - 可选参数：批次大小（用以控制 tokenizer 的批处理）、是否跳过注释/特殊标记、忽略超出模型 vocab_size 的 id、并发 worker 数等。

- 输出：
    - 一个文本文件（例如 `/tmp/custom_static_vocab.txt`），包含最多 k 个不同的 token id；先按在语料中出现频率选出 top-k（去重），然后将这 k 个 token id 按数值升序排序并写入文件（每行一个整数，不带其它注释）。

文件格式与约定
- 文件为 UTF-8 编码的纯文本，每行仅包含一个非负整数，代表模型词汇表中的 token id；例如：

```
3
17
502
...
```

- 如果语料中出现的某些 token id 超出模型 `vocab_size` 范围（例如 token_id >= vocab_size 或 token_id < 0），这些行应被忽略并记录为警告。
- 若语料中不同文本映射到相同 token id（正常情况），计数应合并到该 token id 的总频次中。
- 若语料中有效 token id 总数小于 k，则输出所有可用 token id（按频率排序）。

实现要点（建议）
- 使用模型对应的 HuggingFace tokenizer（或等效的 tokenizer API）将文本分割为 token ids。例如：
    - 调用 tokenizer.encode / tokenizer.__call__（注意设置 return_tensors=None，禁用添加 BOS/EOS，视情况关闭 truncation）。
    - 为避免内存峰值，使用流式读取与分批（batch）分词；对大型语料（如 SlimPajama）建议逐文件、逐行或按固定字节块分割处理。
- 采用高效计数器（例如 Python 的 collections.Counter 或 numpy 聚合）来统计每个 token id 的出现次数；计数过程中跳过非整数或超出范围的 token id。
- 最后基于计数器选择前 k 个 token id（按出现次数降序），若出现次数相同，可按 token id 升序作为次级排序保证确定性。
- 输出时写入临时文件并 atomic rename，避免中途失败导致不完整文件。

CLI 示例（建议）

```
python scripts/generate_static_vocab_from_slimpajama.py \
    --tokenizer-path Qwen/Qwen3-4B \
    --corpus-dir /data/slimpajama \
    --k 32000 \
    --output /tmp/custom_static_vocab.txt \
    --batch-size 8192 \
    --workers 4
```

最小示例（核心步骤）

```python
from collections import Counter
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", use_fast=True)
counter = Counter()

def process_text(text):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        counter.update(ids)

# 对大语料，请以流式方式调用 process_text

# 取 top-k（按频率），再按 token id 升序写出
top_k = [tid for tid, _ in counter.most_common(k)]
# 过滤掉越界 id 并去重（most_common 已按 id 合并计数），然后按数值排序
filtered = [tid for tid in top_k if 0 <= tid < tokenizer.vocab_size]
filtered_sorted = sorted(filtered)
with open("/tmp/custom_static_vocab.txt.tmp", "w", encoding="utf-8") as f:
    for tid in filtered_sorted:
        f.write(f"{tid}\n")
os.replace("/tmp/custom_static_vocab.txt.tmp", "/tmp/custom_static_vocab.txt")

# 如果你已经有一个按频率输出的文件，可以用系统工具数值排序：
# sort -n /tmp/custom_static_vocab.txt -o /tmp/custom_static_vocab.sorted.txt
```

注意与兼容性
- 确保 tokenizer 与目标模型使用相同的 vocab / tokenizer 配置（tokenizer.vocab_size 与模型的 `vocab_size` 一致）以避免 id 对应错误。
- 如果需要对多模型支持，允许传入模型 `vocab_size` 参数以便在写入时进行边界检查。
- 对于多文件和超大语料，建议支持分布式或并行处理以加速统计。

#### 词表精简与验证附加约束（新增）

为保证精简静态词汇表在推测解码与目标模型验证环节的正确性，新增以下约束：

1. 支持乱序的精简词元 ID：
    - 即使静态词汇表文件内的 token id 不是按原始词表序号排列（乱序或稀疏），系统也应当能够正确构建对应的 `lm_head` 子权重并完成 logits 计算。
    - 实现建议：加载精简列表（例如 `[101, 3, 502, 17]`），在内存中同时准备两个映射表：
      - `reduced_index -> original_token_id`（用于把子词表上的采样/索引映射回原始 id），
      - `original_token_id -> reduced_index`（用于在需要将原始 logits 投影到子空间时做快速索引）。

2. 验证时的 ID 映射语义：
    - 如果草稿模型输出以“精简词元的 token id”作为验证依据（即草稿模型在子词表空间内直接返回 id），则在将这些 id 发送给目标模型或用于对比前，必须映射回原始全量词表的 token id。举例：若原始词表被精简为 `[1,3,4]`（文件中按行写为 1、3、4），那么子词表内的索引 `0,1,2` 分别对应原始 id `1,3,4`；如果草稿返回子词表 id `1`（表示第二个条目），验证时应将其映回原始 id `3` 再传给目标模型或做 match。
    - 实现建议：把映射逻辑封装为小工具函数（例如 `map_reduced_ids_to_original(reduced_ids)`），并在验证/比较路径中明确调用以避免混淆。

3. 每次验证记录可审计的日志（放在当前工作目录）：
    - 对每个验证请求（或每个样本）记录一个结构化条目（建议使用 JSONL），字段至少包含：`sample_id`、`prompt`、`draft_generated_tokens`（字符串数组）、`draft_generated_reduced_ids`（子词表 id 数组，如果有）、`draft_generated_original_ids`（映射回的原始 id 数组）、`target_validation_result`（accept/reject）、以及当 `reject` 时 `target_regenerated_tokens` 和对应的 id 列表。
    - 文件命名建议：`validation_log.YYYYMMDD_HHMMSS.jsonl`，便于归档与回溯。
    - 这些日志将帮助诊断是否因为精简词表导致拒绝（即目标模型缺失关键 token）或只是草稿生成不稳定所致。

示例 JSONL 条目（单行）：

```json
{
  "sample_id": 42,
  "prompt": "患者，男，45 岁...",
  "draft_generated_tokens": ["入院", "后"],
  "draft_generated_reduced_ids": [5, 6],
  "draft_generated_original_ids": [101, 502],
  "target_validation_result": "reject",
  "target_regenerated_tokens": ["入院后"],
  "target_regenerated_original_ids": [101, 502]
}
```

上述新约束旨在保证：即便静态词表是经过挑选并且 token id 在文件中是乱序的，整个生成→映射→验证→日志的闭环仍能正确工作，且验证所用的一切 id 都能被映回原词表以避免验证误判。

用途
- 生成的文件可直接作为 `--speculative-static-vocab-path` 或模型配置中的 `custom_vocab_path` 传入，从而在草稿模型中启用词汇表裁剪，显著减少 LM head 的计算量并提高草稿模型速度。

###### **流式下载 & 分片统计 (进一步约束)**

考虑到 SlimPajama 体量极大（数 TB 级），在单机上完整落盘再统计既耗时又占空间。我们要求工具支持“边下载、边分词、边计数”的流式模式，流程如下：

1. **输入**：
   - 分片 URL 列表或脚本可推导出的远程路径（可来自 SlimPajama 官方 manifest）。
   - 草稿模型的分词器（与主模型保持一致）。
   - 可配置的分片大小 / 缓冲区（例如按文件、按 N MB 块）。
   - 目标输出：
       - ① 原始频次文件 `token_freq.txt`（按出现频率降序列出 `token_id\tcount`）。
       - ② 静态词汇表 `custom_static_vocab.txt`（在频次表基础上取前 k 个 token id，再按数值升序写入，每行一个 id）。

2. **处理循环（伪流程）**：

```
for shard in manifest:
    stream = open_remote(shard)        # 使用 HTTP Range / S3 / GCS API
    for payload in chunk(stream, max_bytes=chunk_size):
        texts = parse(payload)         # 解析 JSONL/Parquet 行
        token_ids = tokenizer(texts)   # add_special_tokens=False
        counter.update(token_ids)
    flush_checkpoint(counter)          # 可选：周期性保存中间结果
```

3. **中间状态与容错**：
   - 计数器可每处理 N 行/文件就持久化（例如写入 `counts.temp.json`），以便失败后恢复。
   - 支持断点恢复：记录已完成分片 ID，重启时跳过。

4. **输出阶段**：
   - `token_freq.txt`: `most_common()` 结果；若文件过大，可写入 `token_id\tcount` 并按 count 降序。
   - `custom_static_vocab.txt`: 读取频次文件、取前 k 个、过滤越界 id、按 token id 升序输出（同前述要求）。
   - 可选再提供 `sort -n` 命令或 `heapq.nsmallest`，以便用户自定义排序方式。

5. **CLI 示例**：

```
python scripts/stream_slimpajama_vocab.py \
  --manifest s3://datasets/slimpajama/manifest.json \
  --tokenizer Qwen/Qwen3-4B \
  --chunk-size-mb 128 \
  --topk 32000 \
  --freq-output /tmp/token_freq.txt \
  --vocab-output /tmp/custom_static_vocab.txt \
  --resume-cache /tmp/slim_counter.chkpt
```

这样我们无需一次性下载完整语料，只需保证网络带宽可持续即可。统计结果（频次表 + 排序后的静态词表）可直接复用在草稿模型的静态词汇功能中。


###### **领域语料 Accept Rate 评测（新增）**

除 SlimPajama 外，还需要对**指定的某一份领域语料**（例如医学报告合集）直接做截断补全实验。具体做法：对这份语料里的每一条记录，只截取其中固定比例（默认 50%）作为 prompt，剩余部分作为参考答案；在不额外扩充或打乱语料的前提下，使用启用静态词表的草稿模型执行补写，并统计推测解码 `accept_rate`（可来自 SGLang 日志或 metrics 导出）。

**合同（Contract）**
- 输入：
    - 领域语料文件：推荐 JSONL/CSV/纯文本格式，需包含 `text` 字段（可扩展为多字段）。
    - `truncate_ratio`：截断比例，默认 0.5，表示前半段作为提示，后半段作为 ground truth。
    - 草稿模型/主模型配置（含静态词表文件）。
    - 可选：采样条数、最大生成长度、请求并发度等。
- 输出：
    - 评估报告（CSV/JSON）：记录每条样本的 prompt、参考结尾、模型补写结果、accept_rate（单条或整体统计）。
    - 聚合指标：平均 accept_rate、命中率、拒绝数等，可直接对比不同静态词表策略。

**流程建议**
1. **语料截断**
    - 读取给定语料（可以是单个 JSONL/CSV/纯文本文件），逐条去除空白。
    - 对每条记录直接按 `truncate_ratio` 切分：`prompt = text[:N*ratio]`, `reference = text[N*ratio:]`，无需额外数据增强或混洗。
    - 可将 `prompt/reference` 和 `sample_id` 写入一个简单的 JSONL 方便后续批量评测；如果语料较小，也可在内存中直接构造请求。
2. **调用 SGlang 服务**
     - 启动启用静态词表的草稿模型/主模型：

```
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen3-1.7B \
    --speculative-use-static-vocab \
    --speculative-static-vocab-path /tmp/custom_static_vocab.txt \
    --max-total-tokens 8192 \
    --log-level info
```

3. **批量评测脚本（待实现）**
    - 针对语料中的每条记录（或者抽样 N 条）构造补写请求，`max_new_tokens` 建议设置为 `len(reference)`，保持补全长度一致。
    - 解析响应中的 `accept_rate`（或通过 server metrics）；也可以从 server 的 request log 抓取 `accept_rate` 字段。
    - 将结果写入例如 `domain_eval_result.jsonl`：

```
{
    "sample_id": 42,
    "prompt": "患者，男，45 岁...",
    "reference": "入院后给予...",
    "completion": "入院后给予...",
    "accept_rate": 0.78
}
```

4. **指标汇总**
     - 计算平均 accept_rate / 中位数 / P90，并与完整版词表基线比较。
     - 若 accept_rate 低于阈值，可回溯具体样本分析：是因为静态词表缺失关键 token 还是模型补写偏差。

**CLI 草案**

```
python scripts/domain_corpus_accept_eval.py \
    --corpus medical_notes.jsonl \
    --text-key text \
    --truncate-ratio 0.5 \
    --sglang-endpoint http://127.0.0.1:30000/generate \
    --output reports/medical_accept_eval.jsonl \
    --max-samples 500 \
    --concurrency 8
```

**产出要求**
- 提供可复现的评测脚本/Notebook，附带 README，说明如何准备领域语料、如何设置静态词表以及如何解读 accept_rate。
- 评测报告最好附带可视化（直方图或箱线图）展示 accept_rate 分布，帮助判断静态词表对特定领域的适用性。


此阶段将允许用户提供一个外部文件来定义词汇表，给予用户更大的灵活性。

**2.1. 修改模型初始化逻辑**

我们需要在模型初始化时检查 `custom_vocab_path` 参数。如果该路径有效，则加载文件内容作为词汇表，并优先于第一阶段的比例设置。

**自定义词汇表文件格式** (`custom_vocab.txt`):
一个纯文本文件，每行包含一个词元 ID。
```
101
102
205
...
8034
```

**伪代码/代码片段** (修改 `LlamaForCausalLM.__init__`):

```python
# python/sglang/srt/models/llama.py

import torch
import os

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # ...
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.active_vocab_indices = None # 重命名以反映其通用性

        if config.use_static_vocab:
            vocab_indices = None
            # --- 新增：优先加载自定义词汇表 ---
            if config.custom_vocab_path and os.path.exists(config.custom_vocab_path):
                try:
                    with open(config.custom_vocab_path, 'r') as f:
                        # 从文件加载，并转换为整数
                        token_ids = [int(line.strip()) for line in f if line.strip()]
                    vocab_indices = torch.tensor(token_ids, dtype=torch.long, device="cuda")
                    print(f"Loaded {len(vocab_indices)} tokens from custom vocabulary: {config.custom_vocab_path}")
                except Exception as e:
                    print(f"Warning: Failed to load custom vocabulary file. Error: {e}")

            # --- 如果未加载自定义词汇表，则回退到比例模式 ---
            if vocab_indices is None and 0.0 < config.static_vocab_ratio <= 1.0:
                static_vocab_size = int(config.vocab_size * config.static_vocab_ratio)
                vocab_indices = torch.arange(static_vocab_size, device="cuda")
                print(f"Using static vocabulary with ratio {config.static_vocab_ratio}, size: {static_vocab_size}")

            self.active_vocab_indices = vocab_indices

    def forward(self, ...):
        # ...
        # 将 self.static_vocab_indices 替换为 self.active_vocab_indices
        if self.active_vocab_indices is not None:
            active_lm_head_weight = self.lm_head.weight.index_select(0, self.active_vocab_indices)
            logits = torch.matmul(hidden_states, active_lm_head_weight.t())
        else:
            logits = self.lm_head(hidden_states)

        return logits, ...
```

**关键点**：
*   实现了配置的优先级：`custom_vocab_path` > `static_vocab_ratio`。
*   增加了文件加载和错误处理逻辑。
*   将 `static_vocab_indices` 重命名为更通用的 `active_vocab_indices`，为第三阶段做准备。

---

##### **第三阶段：实现“静态基础+动态扩展”的混合词汇表框架**

这是最核心和最灵活的阶段。我们将创建一个专用的管理器来维护一个可在线更新的词汇表。

**3.1. 设计 `DynamicVocabularyManager` 类**

这个类将是词汇表管理的核心。它应该被设计为线程安全的（如果 SGlang 在多线程环境中使用），并提供清晰的接口。

**伪代码/代码片段** (可以创建一个新文件 `python/sglang/srt/utils/vocabulary_manager.py`):

```python
# python/sglang/srt/utils/vocabulary_manager.py

import torch
from typing import List, Optional

class DynamicVocabularyManager:
    """
    管理一个混合词汇表，由固定的静态基础和可动态扩展的部分组成。
    """
    def __init__(self, base_vocab_indices: Optional[torch.Tensor] = None):
        """
        初始化管理器。

        Args:
            base_vocab_indices (Optional[torch.Tensor]):
                一个一维张量，包含固定的静态词汇表词元 ID。
                在模型初始化时提供。
        """
        # 静态基础词汇表，永不改变
        self.base_vocab = set(base_vocab_indices.tolist() if base_vocab_indices is not None else [])

        # 动态添加的词汇表，可在线更新
        self.dynamic_vocab = set()

        # 当前合并后的活动词汇表（缓存）
        self._active_vocab_cache: Optional[torch.Tensor] = None
        self._is_dirty = True  # 标记缓存是否需要重建

    def add(self, token_ids: List[int]):
        """
        向动态词汇表中添加新的词元 ID。
        这是一个外部接口，可由其他模块（如前缀缓存分析器）调用。

        Args:
            token_ids (List[int]): 要添加的词元 ID 列表。
        """
        new_tokens = set(token_ids)
        # 只有当有新词元加入时，才将缓存标记为“脏”
        if not new_tokens.issubset(self.dynamic_vocab):
            self.dynamic_vocab.update(new_tokens)
            self._is_dirty = True

    def get_active_vocab(self) -> torch.Tensor:
        """
        获取当前完整、去重、排序后的活动词汇表索引。
        如果缓存有效，则直接返回；否则，重建缓存。

        Returns:
            torch.Tensor: 一个包含所有活动词汇表词元 ID 的一维张量。
        """
        if self._is_dirty or self._active_vocab_cache is None:
            # 合并静态和动态词汇表
            merged_vocab = sorted(list(self.base_vocab.union(self.dynamic_vocab)))
            self._active_vocab_cache = torch.tensor(merged_vocab, dtype=torch.long, device="cuda")
            self._is_dirty = False

        return self._active_vocab_cache

    def get_active_vocab_size(self) -> int:
        """返回当前活动词汇表的大小。"""
        return len(self.get_active_vocab())

```

**3.2. 集成 `DynamicVocabularyManager` 到草稿模型**

现在，我们将用这个管理器来替换之前简单的 `active_vocab_indices`。

**伪代码/代码片段** (再次修改 `LlamaForCausalLM`):

```python
# python/sglang/srt/models/llama.py

# from sglang.srt.utils.vocabulary_manager import DynamicVocabularyManager # 导入新类

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # ...
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # --- 初始化 DynamicVocabularyManager ---
        self.vocab_manager = None
        if config.use_static_vocab:
            base_indices = None
            # (与第二阶段相同的逻辑来加载静态基础词汇表)
            if config.custom_vocab_path and os.path.exists(config.custom_vocab_path):
                # ... load from file ...
                base_indices = torch.tensor(...)
            elif 0.0 < config.static_vocab_ratio <= 1.0:
                # ... create from ratio ...
                base_indices = torch.arange(...)

            # 创建管理器实例
            self.vocab_manager = DynamicVocabularyManager(base_vocab_indices=base_indices)

    def forward(self, input_ids, position_ids, seq_lens, ..., new_dynamic_tokens: Optional[List[int]] = None):
        # ...

        # --- 新增：在每次 forward 调用时，动态更新词汇表 ---
        if self.vocab_manager and new_dynamic_tokens:
            self.vocab_manager.add(new_dynamic_tokens)

        # --- 修改 Logits 计算逻辑以使用管理器 ---
        if self.vocab_manager:
            # 1. 在每次推理时获取最新的活动词汇表
            active_vocab_indices = self.vocab_manager.get_active_vocab()

            # 2. 在此动态变化的词汇表上计算 Logits
            active_lm_head_weight = self.lm_head.weight.index_select(0, active_vocab_indices)
            logits = torch.matmul(hidden_states, active_lm_head_weight.t())

            # 3. (重要) 后续的采样和 token 映射逻辑需要使用这个动态的 active_vocab_indices
        else:
            logits = self.lm_head(hidden_states)

        return logits, ...
```

**3.3. 设计外部调用接口**

`DynamicVocabularyManager.add()` 方法是留给外部逻辑调用的。例如，一个分析前缀缓存（Prefix Cache）的模块可以在每次迭代后，识别出新的、可能高频的词元，并通过这个接口将其添加到动态词汇表中。

```python
# 示例：在一个推理协调器或管理器中调用
def inference_step(...):
    # ...
    # 运行草稿模型
    draft_outputs = draft_model.forward(...)

    # 假设我们有一个分析器，它从最近的上下文中提取了新的高频词元
    newly_discovered_tokens = analyze_context_for_new_tokens(context) # e.g., [50257, 198, 628]

    # 将新发现的词元添加到下一次迭代的词汇表中
    # 注意：这里需要将 new_dynamic_tokens 传递给下一次的 forward 调用
    # 或者，如果 draft_model 是一个共享实例，可以直接调用
    # draft_model.vocab_manager.add(newly_discovered_tokens)
```

**关键点**：
*   **模块化设计**：`DynamicVocabularyManager` 将词汇表管理的复杂性封装起来，使模型代码更清晰。
*   **高效缓存**：通过 `_is_dirty` 标志和 `_active_vocab_cache`，避免了在词汇表未变化时不必要的合并、排序和张量创建操作。
*   **动态性**：`forward` 方法现在能够在每次调用时适应一个动态变化的词汇表，这是实现自适应优化的基础。
*   **可扩展性**：`add` 接口提供了一个清晰的扩展点，未来可以接入更复杂的词元发现策略（如基于梯度的词元选择、n-gram 分析等）。

---

#### **4. 总结与后续工作**

本文档详细描述了为 SGlang 实现三阶段词汇表优化功能的开发流程。按照这个指南，您可以逐步构建一个功能完善、模块化且可扩展的系统。

**最终框架概览**：
1.  **配置层** (`ModelConfig`)：通过 `use_static_vocab`, `static_vocab_ratio`, `custom_vocab_path` 控制功能的开启和基础词汇表的来源。
2.  **管理层** (`DynamicVocabularyManager`)：封装静态与动态词汇表的合并、缓存和更新逻辑。
3.  **执行层** (`LlamaForCausalLM.forward`)：在每次推理时，从管理层获取最新的活动词汇表，并在此子集上执行计算密集型的操作。

**后续建议**：
*   **性能分析**：在实现后，对不同词汇表大小下的推理速度和内存占用进行详细的基准测试。
*   **精度评估**：评估词汇表裁剪对草稿模型准确率（即推测命中率）的影响，找到速度和精度之间的最佳平衡点。
*   **实现动态词元发现逻辑**：开发一个或多个策略来调用 `DynamicVocabularyManager.add()`，例如基于近期生成历史、注意力分数或特定任务上下文来动态添加词元。

希望这份文档能为您的开发工作提供清晰的指引。
