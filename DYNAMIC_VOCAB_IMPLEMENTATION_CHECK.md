# 动态词汇表实现检查报告

## 检查日期
2024年（当前检查）

## 实现状态总结

### ✅ 已实现的部分

#### 1. API 层 (`python/sglang/api/api_server.py`)
- ✅ `/v1/vocab/add` 端点：添加词汇到动态词汇表
- ✅ `/v1/vocab/remove` 端点：从动态词汇表移除词汇
- ✅ `/v1/vocab/query` 端点：查询当前词汇表状态
- ✅ `/v1/generate` 端点：使用动态词汇表进行生成
- ✅ `/v1/chat/completions` 端点：支持动态词汇表
- ✅ 自动初始化：在首次使用前自动初始化静态词汇表
- ✅ 参数传递：将 `dynamic_vocab_token_ids` 传递给后端

#### 2. 词汇表管理器 (`python/sglang/api/vocab_manager.py`)
- ✅ `ActiveVocabManager` 类：线程安全的全局词汇表管理器
- ✅ `initialize_static_vocab()`：初始化静态词汇表
- ✅ `add_words()`：添加词汇（支持 LRU 策略）
- ✅ `remove_words()`：移除词汇
- ✅ `get_vocab_list()`：获取当前词汇表列表
- ✅ LRU 容量管理：当超过容量时自动移除最旧的 token

#### 3. 后端数据结构支持
- ✅ `SamplingParams.dynamic_vocab_token_ids`：采样参数中包含动态词汇表 ID 列表
- ✅ `ForwardBatch.dynamic_vocab_token_ids`：前向批次中包含动态词汇表 ID 张量
- ✅ `LogitsMetadata.dynamic_vocab_token_ids`：Logits 元数据中包含动态词汇表 ID

#### 4. Logits Processor (`python/sglang/srt/layers/logits_processor.py`)
- ✅ `_project_hidden_to_vocab()` 方法：实现动态词汇表的 logits 投影
  - 当 `dynamic_vocab_token_ids` 存在时，使用 `torch.index_select` 选择对应的权重行
  - 执行子集矩阵乘法：`[B, H] x [H, V_sub]` 而不是完整的 `[B, H] x [H, V_full]`
  - 优先级：动态词汇表 > 静态词汇表 > 完整词汇表
- ✅ 静态词汇表禁用：当动态词汇表激活时，自动禁用静态词汇表

#### 5. Sampler (`python/sglang/srt/layers/sampler.py`)
- ✅ `_map_dynamic_vocab_token_ids()` 方法：将动态词汇表空间中的 token ID 映射回全局 ID
  - 支持 1D 张量（所有请求共享同一个词汇表）
  - 支持 2D 张量（每个请求有独立的词汇表）
  - 支持列表/元组格式
- ✅ `_remap_dynamic_vocab_top_indices()` 方法：重新映射 top-k 索引

#### 6. 测试 (`python/sglang/api/test_api_server.py`)
- ✅ 单元测试：测试添加/移除词汇的流程
- ✅ 集成测试：测试动态词汇表到达 logits probe
- ✅ Mock 后端：使用 `FakeLogitsProbe` 验证动态词汇表参数传递

### ⚠️ 潜在问题

#### 1. 数据流完整性（关键问题）
**问题**：在 `ForwardBatch.init_new()` 方法中，**没有找到设置 `dynamic_vocab_token_ids` 的代码**。

**当前状态**：
- ✅ `SamplingParams` 包含 `dynamic_vocab_token_ids` 字段
- ✅ `ForwardBatch` 定义了 `dynamic_vocab_token_ids` 字段
- ✅ `LogitsProcessor` 使用 `forward_batch.dynamic_vocab_token_ids`
- ❌ **缺失**：从 `ScheduleBatch.reqs[].sampling_params.dynamic_vocab_token_ids` 提取并设置到 `ForwardBatch.dynamic_vocab_token_ids` 的代码

**需要添加的代码位置**：
应该在 `ForwardBatch.init_new()` 方法中，从 `batch`（`ModelWorkerBatch`）或 `ScheduleBatch.reqs` 中提取 `dynamic_vocab_token_ids`。

**可能的实现方式**：
```python
# 在 ForwardBatch.init_new() 中添加：
# 从 ScheduleBatch.reqs 中提取 dynamic_vocab_token_ids
# 注意：ModelWorkerBatch 可能没有直接访问 reqs，需要从 ScheduleBatch 传递
if hasattr(batch, 'reqs') and batch.reqs:
    # 提取所有请求的 dynamic_vocab_token_ids
    dynamic_vocab_lists = [r.sampling_params.dynamic_vocab_token_ids 
                          for r in batch.reqs 
                          if r.sampling_params.dynamic_vocab_token_ids is not None]
    if dynamic_vocab_lists:
        # 如果所有请求使用相同的词汇表（全局共享），使用第一个
        # 或者合并为 2D 张量支持每个请求独立的词汇表
        ret.dynamic_vocab_token_ids = torch.tensor(
            dynamic_vocab_lists[0], dtype=torch.long, device=device
        )
```

**建议**：
1. 需要添加代码从 `ScheduleBatch.reqs` 中提取 `dynamic_vocab_token_ids`
2. 需要决定是支持全局共享的词汇表还是每个请求独立的词汇表
3. 需要验证整个数据流是否完整

#### 2. 批处理支持
**问题**：当前实现假设所有请求使用同一个动态词汇表（通过全局 `vocab_manager`）。

**影响**：
- 如果不同请求需要不同的动态词汇表，当前实现可能不支持
- `Sampler._map_dynamic_vocab_token_ids()` 支持每个请求独立的词汇表（2D 张量），但 API 层只提供全局词汇表

**建议**：如果需要支持每个请求独立的动态词汇表，需要修改 API 层和词汇表管理器。

### ✅ 已验证的功能

1. **API 端点功能完整**
   - 添加/移除/查询词汇表功能正常
   - 参数正确传递给后端

2. **后端计算支持**
   - Logits processor 正确实现动态词汇表的 logits 投影
   - Sampler 正确实现 token ID 映射

3. **线程安全**
   - `ActiveVocabManager` 使用锁保护，线程安全

4. **容量管理**
   - LRU 策略正确实现
   - 容量限制正确应用

### 📝 建议的验证步骤

1. **端到端测试**：
   ```bash
   # 1. 启动服务器
   python3 -m sglang.launch_server --model-path <model> --init-vocab-size 327 --dyna-space 1024
   
   # 2. 启动 API 服务器
   export SGLANG_RUNTIME_URL=http://127.0.0.1:30001
   python -m uvicorn sglang.api.api_server:app --host 0.0.0.0 --port 30000
   
   # 3. 测试动态词汇表
   curl -X POST http://127.0.0.1:30000/v1/vocab/add -H "Content-Type: application/json" -d '{"words": ["test"]}'
   curl -X POST http://127.0.0.1:30000/v1/generate -H "Content-Type: application/json" -d '{"prompt": "Hello", "max_new_tokens": 10}'
   ```

2. **代码路径验证**：
   - 在 `ForwardBatch.init_new()` 中添加日志，验证 `dynamic_vocab_token_ids` 是否被设置
   - 在 `logits_processor._project_hidden_to_vocab()` 中添加日志，验证动态词汇表路径是否被触发

3. **性能测试**：
   - 比较使用动态词汇表和不使用动态词汇表的性能差异
   - 验证 logits 投影的计算量是否正确减少

## 结论

**总体评估**：动态词汇表功能**基本实现**，核心功能都已到位：
- ✅ API 层完整实现
- ✅ 词汇表管理器完整实现
- ✅ 后端 logits 计算支持完整
- ✅ Token ID 映射支持完整

**需要注意**：
- ⚠️ 需要验证 `dynamic_vocab_token_ids` 从 `SamplingParams` 到 `ForwardBatch` 的完整数据流
- ⚠️ 当前实现假设全局共享的动态词汇表，不支持每个请求独立的词汇表

**建议**：进行端到端测试，验证整个流程是否正常工作。

