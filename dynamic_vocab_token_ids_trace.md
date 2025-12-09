# `logits_metadata.dynamic_vocab_token_ids` 传入路径追踪

本文档追踪 `logits_metadata.dynamic_vocab_token_ids` 从 API 层到 LogitsProcessor 的完整传入路径。

## 调用链概览

```
API Server (api_server.py)
  ↓
SamplingParams (sampling_params.py)
  ↓
ScheduleBatch (schedule_batch.py)
  ↓
ModelWorkerBatch (schedule_batch.py)
  ↓
ForwardBatch (forward_batch_info.py)
  ↓
LogitsMetadata (logits_processor.py)
  ↓
LogitsProcessor (logits_processor.py)
```

## 详细代码路径

### 1. API 层：设置到 sampling_params

**文件**: `python/sglang/api/api_server.py`

**位置 1**: `/v1/generate` 端点 (第 574 行)
```python
@app.post("/v1/generate")
def generate(req: GenerateRequest):
    effective_client_id = _resolve_client_id(req.client_id)
    _ensure_client_vocab_initialized(effective_client_id)
    active_vocab_ids = vocab_manager.get_vocab_list()
    if not active_vocab_ids:
        raise HTTPException(
            status_code=400,
            detail=f"No active vocabulary for client {effective_client_id}",
        )

    sampling_kwargs = dict(req.sampling_params or {})
    sampling_kwargs["dynamic_vocab_token_ids"] = active_vocab_ids  # ← 第 574 行
    # ...
    result = backend.generate(
        req.prompt,
        sampling_params=sampling_kwargs,
    )
```

**位置 2**: `/v1/hidden_states` 端点 (第 616 行)
```python
@app.post("/v1/hidden_states")
def get_hidden_states(req: HiddenStateRequest):
    effective_client_id = _resolve_client_id(req.client_id)
    _ensure_client_vocab_initialized(effective_client_id)
    active_vocab_ids = vocab_manager.get_vocab_list()
    # ...
    sampling_kwargs["dynamic_vocab_token_ids"] = active_vocab_ids  # ← 第 616 行
```

**位置 3**: `/v1/chat/completions` 端点 (第 692 行)
```python
@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    # ...
    if req.client_id:
        _ensure_client_vocab_initialized(req.client_id)
        active_vocab_ids = vocab_manager.get_vocab_list()
        if active_vocab_ids:
            sampling_kwargs["dynamic_vocab_token_ids"] = active_vocab_ids  # ← 第 692 行
```

### 2. SamplingParams：存储参数

**文件**: `python/sglang/srt/sampling/sampling_params.py`

**位置**: `SamplingParams.__init__` (第 62, 93 行)
```python
class SamplingParams:
    def __init__(
        self,
        # ... 其他参数 ...
        dynamic_vocab_token_ids: Optional[List[int]] = None,  # ← 第 62 行
    ) -> None:
        # ...
        # The active subset of vocab IDs to use for dynamic vocab projection.
        self.dynamic_vocab_token_ids = dynamic_vocab_token_ids  # ← 第 93 行
```

### 3. ScheduleBatch：从请求提取并转换为 Tensor

**文件**: `python/sglang/srt/managers/schedule_batch.py`

**位置**: `ScheduleBatch.make_model_worker_batch` (第 1820-1834 行)
```python
def make_model_worker_batch(self, ...) -> ModelWorkerBatch:
    # ...
    
    # Extract dynamic_vocab_token_ids from requests
    # All requests share the same global dynamic vocab (1D tensor only)
    dynamic_vocab_token_ids = None
    if self.reqs:
        # Get dynamic_vocab_token_ids from the first request
        # All requests should use the same global dynamic vocab
        first_req = self.reqs[0]
        if first_req.sampling_params.dynamic_vocab_token_ids is not None:  # ← 第 1827 行
            vocab_list = first_req.sampling_params.dynamic_vocab_token_ids
            if isinstance(vocab_list, list):
                dynamic_vocab_token_ids = torch.tensor(vocab_list, dtype=torch.long)  # ← 第 1830 行
            elif isinstance(vocab_list, torch.Tensor):
                dynamic_vocab_token_ids = vocab_list.to(torch.long)  # ← 第 1832 行
            else:
                dynamic_vocab_token_ids = torch.tensor(list(vocab_list), dtype=torch.long)  # ← 第 1834 行

    return ModelWorkerBatch(
        # ... 其他字段 ...
        dynamic_vocab_token_ids=dynamic_vocab_token_ids,  # ← 第 1884 行
    )
```

### 4. ModelWorkerBatch：定义字段

**文件**: `python/sglang/srt/managers/schedule_batch.py`

**位置**: `ModelWorkerBatch` 类定义 (第 1999-2004 行)
```python
class ModelWorkerBatch:
    # ... 其他字段 ...
    
    # Dynamic vocab (subset projection) support.
    # This can be:
    # - None: no dynamic vocab
    # - 1D tensor: all requests share the same dynamic vocab (global shared)
    # - 2D tensor: each request has its own dynamic vocab (shape: [batch_size, vocab_size])
    dynamic_vocab_token_ids: Optional[torch.Tensor] = None  # ← 第 2004 行
```

### 5. ForwardBatch：从 ModelWorkerBatch 提取并转移到设备

**文件**: `python/sglang/srt/model_executor/forward_batch_info.py`

**位置**: `ForwardBatch.from_model_worker_batch` (第 390-394 行)
```python
@classmethod
def from_model_worker_batch(
    cls,
    batch: ModelWorkerBatch,
    model_runner: "ModelRunner",
) -> "ForwardBatch":
    # ...
    
    # Extract dynamic_vocab_token_ids from ModelWorkerBatch
    if batch.dynamic_vocab_token_ids is not None:  # ← 第 391 行
        ret.dynamic_vocab_token_ids = batch.dynamic_vocab_token_ids.to(
            device, non_blocking=True  # ← 第 392-393 行
        )
```

**ForwardBatch 字段定义** (第 222 行):
```python
@dataclasses.dataclass
class ForwardBatch:
    # ... 其他字段 ...
    dynamic_vocab_token_ids: Optional[torch.Tensor] = None  # ← 第 222 行
```

### 6. LogitsMetadata：从 ForwardBatch 传入

**文件**: `python/sglang/srt/layers/logits_processor.py`

**位置 1**: `LogitsMetadata.from_forward_batch` (第 182 行)
```python
@classmethod
def from_forward_batch(cls, forward_batch: ForwardBatch):
    # ...
    return cls(
        forward_mode=forward_batch.forward_mode,
        capture_hidden_mode=forward_batch.capture_hidden_mode,
        next_token_logits_buffer=forward_batch.next_token_logits_buffer,
        dynamic_vocab_token_ids=forward_batch.dynamic_vocab_token_ids,  # ← 第 182 行
        # ... 其他字段 ...
    )
```

**LogitsMetadata 字段定义** (第 114 行):
```python
@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    next_token_logits_buffer: Optional[torch.Tensor] = None

    # Dynamic vocab (subset projection) support. Mirrors ForwardBatch.dynamic_vocab_token_ids.
    dynamic_vocab_token_ids: Optional[torch.Tensor] = None  # ← 第 114 行
```

**位置 2**: `LogitsProcessor.forward` (第 453 行)
```python
def forward(
    self,
    input_ids,
    hidden_states,
    lm_head: VocabParallelEmbedding,
    logits_metadata: Union[LogitsMetadata, ForwardBatch],
    aux_hidden_states: Optional[torch.Tensor] = None,
) -> LogitsProcessorOutput:
    if isinstance(logits_metadata, ForwardBatch):
        logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)  # ← 第 453 行
    # ...
```

### 7. LogitsProcessor：使用 dynamic_vocab_token_ids

**文件**: `python/sglang/srt/layers/logits_processor.py`

**位置 1**: `_get_logits` 方法 (第 946-956 行)
```python
def _get_logits(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    logits_metadata: LogitsMetadata,
    embedding_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # ...
    
    # Dynamic vocab should only be used for draft models.
    # For target models, always use full vocabulary.
    is_draft_model = logits_metadata.forward_mode.is_draft_extend(include_v2=True)
    dynamic_vocab_active = (
        logits_metadata.dynamic_vocab_token_ids is not None and is_draft_model  # ← 第 947 行
    )
    # ...
```

**位置 2**: `_project_hidden_to_vocab` 方法 (第 1071-1098 行)
```python
def _project_hidden_to_vocab(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    logits_metadata: LogitsMetadata,
    embedding_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    # Dynamic vocab (subset projection) should only be used for draft models.
    # Target models should use the full vocabulary to ensure output quality.
    dynamic_ids = logits_metadata.dynamic_vocab_token_ids  # ← 第 1071 行
    is_draft_model = (
        logits_metadata.forward_mode.is_draft_extend(include_v2=True)
    )
    
    # Only apply dynamic vocab for draft models, not for target models
    if dynamic_ids is not None and hasattr(lm_head, "weight") and is_draft_model:  # ← 第 1077 行
        # Normalize ids to a LongTensor on the lm_head weight device.
        if not isinstance(dynamic_ids, torch.Tensor):
            dynamic_ids = torch.as_tensor(dynamic_ids, dtype=torch.long)
        else:
            dynamic_ids = dynamic_ids.to(dtype=torch.long)
        dynamic_ids = dynamic_ids.to(lm_head.weight.device)

        proj_dtype = torch.float32 if self.use_fp32_lm_head else lm_head.weight.dtype
        hidden_proj = hidden_states.to(proj_dtype)
        weight = lm_head.weight.to(proj_dtype)
        
        # dynamic_vocab_token_ids is always 1D (global shared vocab)
        # Gather a subset of vocab rows and do a smaller matmul: [B, H] x [H, V_sub]
        if dynamic_ids.dim() != 1:
            raise ValueError(
                f"dynamic_vocab_token_ids must be 1D tensor, got {dynamic_ids.dim()}D"
            )
        weight_slice = torch.index_select(weight, 0, dynamic_ids)  # ← 第 1095 行
        logits = torch.matmul(hidden_proj, weight_slice.T)  # ← 第 1096 行
        
        return logits.to(hidden_states.dtype)
    # ...
```

**位置 3**: 返回给调用者 (第 436, 667 行)
```python
# 在 compute_logprobs_for_multi_item_scoring 中
return LogitsProcessorOutput(
    next_token_logits=None,
    static_vocab_token_ids=self._active_static_indices,
    dynamic_vocab_token_ids=logits_metadata.dynamic_vocab_token_ids,  # ← 第 436 行
    # ...
)

# 在 forward 方法中
return LogitsProcessorOutput(
    next_token_logits=sampled_logits,
    static_vocab_token_ids=self._active_static_indices,
    dynamic_vocab_token_ids=logits_metadata.dynamic_vocab_token_ids,  # ← 第 667 行
    # ...
)
```

## 总结

`logits_metadata.dynamic_vocab_token_ids` 的传入路径：

1. **API 层** (`api_server.py`): 从 `vocab_manager.get_vocab_list()` 获取，设置到 `sampling_kwargs["dynamic_vocab_token_ids"]`
2. **SamplingParams** (`sampling_params.py`): 作为参数存储
3. **ScheduleBatch** (`schedule_batch.py`): 从请求的 `sampling_params` 提取，转换为 `torch.Tensor`，传递给 `ModelWorkerBatch`
4. **ModelWorkerBatch** (`schedule_batch.py`): 存储为字段
5. **ForwardBatch** (`forward_batch_info.py`): 从 `ModelWorkerBatch` 提取，转移到目标设备
6. **LogitsMetadata** (`logits_processor.py`): 通过 `from_forward_batch()` 从 `ForwardBatch` 传入
7. **LogitsProcessor** (`logits_processor.py`): 在 `_project_hidden_to_vocab()` 中使用，进行动态词汇表投影

