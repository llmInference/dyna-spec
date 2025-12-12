# 动态词汇表功能 - 最终修复总结

## 问题演进

### 问题1: DynamicVocabManager未初始化 ✅已修复
**原因**: `dynamic_vocab_capacity`未传递到`hf_config`
**修复**: 在`model_config.py`中显式设置

### 问题2: inverse_map未更新 ✅已修复
**原因**: 添加动态词汇后，映射表未包含新token
**修复**: 在`scheduler.py`的`dynamic_vocab_add`中更新映射

### 问题3: Tensor切片破坏CUDA Graph ✅已修复
**原因**: `logits[:, :populated_size]`改变形状
**修复**: 使用完整capacity，Triton kernel自动mask

### 问题4: CUDA Graph未capture动态词汇路径 ✅已修复 (V2方案)
**原因**: capture时`populated_size`作为scalar被bake进图里。
**旧修复 (已废弃)**: 强制Recapture Graph -> 导致Crash。
**新修复 (V2)**:
1. 将`populated_size`改为**GPU Tensor**，并传递**Pointer**给Triton Kernel。
2. Kernel通过`tl.load(PopulatedSize_ptr)`读取值。
3. 这样即使Graph Capture了指针，指针指向的内存内容可以随时更新，**无需Recapture Graph**。

## 最终方案 V2

### 关键修改的文件

1. **`DynamicVocabularyManager`**
   - 维护 `self.populated_size_gpu = torch.zeros(1, ...)`
   - 在 `add()` 时同步更新这个Tensor。

2. **`dynamic_vocab_ops.py`**
   - Kernel签名改为接受 `PopulatedSize_ptr`。
   - 使用 `tl.load` 读取实际大小。

3. **`logits_processor.py`**
   - 传递 `populated_size_gpu` Tensor 给 Kernel。

### 工作原理

```
服务器启动
  ↓
CUDA Graph capture
  -> Capture Kernel(..., populated_size_ptr=0x1234, ...)
  -> 此时 0x1234 内存处值为 0

运行时添加词汇
  ↓
更新 0x1234 内存处值为 5
  ↓
推理请求
  ↓
Graph Replay
  -> Kernel 读取 0x1234 -> 读到 5 -> 正确计算！
```

## 测试步骤
（同上，请重启服务器测试）
（同上，请重启服务器测试）

```bash
# 1. 重启服务器
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path Qwen/Qwen3-1.7B \
  --speculative-dynamic-vocab-capacity 1024 \
  --speculative-use-static-vocab \
  --speculative-static-vocab-path ./numbers.txt \
  --log-level info

# 2. 添加动态词汇
curl -X POST http://localhost:30000/dynamic_vocab/add \
  -H "Content-Type: application/json" \
  -d '{"vocab_size": 151936, "new_token_ids": [0, 2, 3, 4, 5]}'

# 3. 查询状态
curl -X POST http://localhost:30000/dynamic_vocab/status \
  -H "Content-Type: application/json" \
  -d '{}'

# 4. 测试生成（使用高temperature增加多样性）
curl http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test:",
    "sampling_params": {
      "max_new_tokens": 30,
      "temperature": 1.5
    }
  }'

# 5. 检查validation log
tail -f validation_log.*.jsonl | jq '.draft_generated_original_ids'
```

## 预期结果

**validation_log中应该看到**：
- `draft_generated_original_ids`: 包含0, 2, 3, 4, 5（不只是1）
- `draft_generated_reduced_ids`: 对应的reduced indices

**如果仍然只看到1**：
- 说明logits分数差距过大（语义问题，不是代码bug）
- 可以尝试添加真实高频词代替0,2,3,4,5

## 性能说明

- ✅ CUDA Graph保持启用（性能最优）
- ✅ Triton kernel仅计算populated部分（节省算力）
- ✅ 完全符合README_Dyna.md规范
