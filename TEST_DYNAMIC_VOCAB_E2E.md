# 动态词汇表端到端测试指南

## 概述

本文档说明如何运行端到端测试来验证动态词汇表功能是否正常工作。

## 前置条件

1. **已安装依赖**：确保在 `sglang2` conda 环境中
2. **模型可用**：需要有可用的模型路径（例如 `Qwen/Qwen3-4B` 和 `Qwen/Qwen3-1.7B`）

## 测试步骤

### 步骤 1：启动 SGLang Runtime 服务器

在第一个终端窗口中运行：

```bash
cd /home/llminference/syq/dyna-spec
source /home/llminference/miniconda3/bin/activate sglang2

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

**注意**：根据你的实际模型路径调整 `--model-path` 和 `--speculative-draft-model-path` 参数。

### 步骤 2：启动 FastAPI 服务器

在第二个终端窗口中运行：

```bash
cd /home/llminference/syq/dyna-spec
source /home/llminference/miniconda3/bin/activate sglang2

export SGLANG_RUNTIME_URL=http://127.0.0.1:30001
python -m uvicorn sglang.api.api_server:app --host 0.0.0.0 --port 30000
```

### 步骤 3：运行端到端测试

在第三个终端窗口中运行：

```bash
cd /home/llminference/syq/dyna-spec
source /home/llminference/miniconda3/bin/activate sglang2

python test_dynamic_vocab_e2e.py
```

## 测试内容

测试脚本会执行以下测试：

1. **词汇表管理功能**
   - 查询初始词汇表状态
   - 添加词汇到动态词汇表
   - 验证词汇表大小变化
   - 从动态词汇表移除词汇

2. **生成功能**
   - 添加测试词汇
   - 使用动态词汇表进行文本生成
   - 验证生成是否成功

3. **多请求共享词汇表**
   - 验证多个请求是否共享同一个词汇表
   - 测试词汇表的全局共享特性

## 预期结果

如果所有测试通过，你应该看到：

```
============================================================
测试总结
============================================================

✅ 通过: 词汇表管理
✅ 通过: 生成功能
✅ 通过: 多请求共享

总计: 3/3 测试通过

🎉 所有测试通过！动态词汇表功能正常工作。
```


## 手动测试

如果你想手动测试，可以使用 curl 命令：

### 1. 查询词汇表状态

```bash
curl -X POST http://127.0.0.1:30000/v1/vocab/query \
  -H "Content-Type: application/json" \
  -d '{"client_id": "test_client"}'
```

### 2. 添加词汇

```bash
curl -X POST http://127.0.0.1:30000/v1/vocab/add \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "test_client",
    "words": ["thing", "example", "test"]
  }'
```

### 3. 使用动态词汇表生成

```bash
curl -X POST http://127.0.0.1:30000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "test_client",
    "prompt": "Hello, world",
    "max_new_tokens": 10,
    "temperature": 0.7
  }'
```

## 验证数据流完整性

测试脚本会验证以下数据流：

1. ✅ API 层：`/v1/vocab/add` 添加词汇到 `vocab_manager`
2. ✅ API 层：`/v1/generate` 从 `vocab_manager` 获取词汇表并传递给后端
3. ✅ 后端：`ScheduleBatch.get_model_worker_batch()` 提取 `dynamic_vocab_token_ids`
4. ✅ 后端：`ForwardBatch.init_new()` 设置 `dynamic_vocab_token_ids`
5. ✅ 后端：`LogitsProcessor._project_hidden_to_vocab()` 使用动态词汇表进行投影

如果所有测试通过，说明数据流是完整的，动态词汇表功能正常工作。

## 性能验证

要验证性能优化是否生效，可以：

1. **监控 logits 计算量**：在 `LogitsProcessor._project_hidden_to_vocab()` 中添加日志
2. **比较计算时间**：比较使用动态词汇表和不使用动态词汇表的生成时间
3. **检查内存使用**：验证 logits 张量的大小是否正确减少

## 相关文档

- `DYNAMIC_VOCAB_IMPLEMENTATION_CHECK.md` - 实现检查报告
- `DYNAMIC_VOCAB_ISSUES_IMPACT.md` - 潜在问题影响分析
- `client/DYNAMIC_VOCAB_SETUP.md` - 配置指南
- `verify_dynamic_vocab.py` - 代码路径验证脚本

