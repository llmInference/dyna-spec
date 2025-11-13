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

##### **第二阶段：支持自定义静态词汇表**

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
