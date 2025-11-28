import os
import threading
import time
import uuid
from typing import List, Optional, Union, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sglang as sgl
from .vocab_manager import vocab_manager

app = FastAPI()

_RUNTIME_URL_ENV = "SGLANG_RUNTIME_URL"
_RUNTIME_API_KEY_ENV = "SGLANG_RUNTIME_API_KEY"
_RUNTIME_VERIFY_ENV = "SGLANG_RUNTIME_VERIFY"
_RUNTIME_TIMEOUT_ENV = "SGLANG_RUNTIME_TIMEOUT"
_TOKENIZER_OVERRIDE_ENV = "SGLANG_TOKENIZER_PATH"
_DEFAULT_HTTP_TIMEOUT = 30.0

_http_backend_lock = threading.Lock()
_http_backend: Optional["_HttpRuntimeBackend"] = None
DEFAULT_CLIENT_ID = "default_client"


class VocabRequest(BaseModel):
    client_id: Optional[str] = None
    words: List[str]


class VocabQueryRequest(BaseModel):
    client_id: Optional[str] = None


class CustomVocabRequest(BaseModel):
    """Request to load a custom vocabulary from token IDs.
    
    This is useful for loading vocabularies obtained from tokenizing datasets
    with the same model's tokenizer. The custom vocabulary will be used for
    the draft model only, while the target model uses the full vocabulary.
    """
    client_id: Optional[str] = None
    token_ids: List[int]
    dyna_space: Optional[int] = None  # Optional dynamic space buffer


class GenerateRequest(BaseModel):
    client_id: Optional[str] = None
    prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    sampling_params: Optional[Dict[str, Any]] = None


class HiddenStateRequest(GenerateRequest):
    """Request payload for fetching hidden states."""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    client_id: Optional[str] = None  # Optional client_id for dynamic vocab


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


def _ensure_tokenizer():
    tokenizer = None
    try:
        tokenizer = sgl.get_tokenizer()
    except Exception:
        tokenizer = None

    if tokenizer is None:
        backend = _get_http_backend()
        if backend is not None:
            try:
                tokenizer = backend.get_tokenizer()
            except Exception as exc:  # pragma: no cover - fallback path
                raise HTTPException(
                    status_code=500, detail=f"Failed to initialize tokenizer: {exc}"
                ) from exc

    if tokenizer is None:
        source_hint = os.environ.get(_TOKENIZER_OVERRIDE_ENV) or "runtime backend"
        raise HTTPException(
            status_code=500,
            detail=(
                "Tokenizer not initialized. Configure a default backend via "
                "sglang.set_default_backend(...) or set "
                f"{_RUNTIME_URL_ENV} and optionally {_TOKENIZER_OVERRIDE_ENV} "
                f"to point at an existing runtime ({source_hint})."
            ),
        )

    return tokenizer


def _ensure_client_vocab_initialized(client_id: str) -> None:
    """Ensure the draft model vocabulary is initialized with the initial vocabulary.
    
    Two initialization modes are supported:
    1. --init-vocab-size mode: Initialize with consecutive token IDs [0, 1, 2, ..., init_vocab_size-1]
    2. --custom-vocab mode: Initialize with custom token IDs from the server
    
    The vocabulary is global and shared across all clients.
    vocab_manager will manage all vocabulary data uniformly, and vocab_count/initial_vocab_count
    can be read directly from vocab_manager without distinguishing modes.
    """
    if not vocab_manager.is_initialized():
        dyna_space = 1024  # Default value
        static_vocab_indices = None
        init_vocab_size = None
        
        # Priority 1: Try to get from SGLang server via /get_model_info
        import logging
        try:
            backend = _ensure_backend()
            if hasattr(backend, "_get_model_info"):
                model_info = backend._get_model_info(force_refresh=True)
                
                # Get dyna_space
                if "dyna_space" in model_info:
                    dyna_space = int(model_info["dyna_space"])
                
                # Determine vocabulary initialization mode based on server response
                # We use static_vocab_indices (not custom_vocab_path) because:
                # 1. custom_vocab_path is a server startup parameter, not available in API server
                # 2. static_vocab_indices is the actual loaded data, more reliable
                # 3. static_vocab_indices presence indicates custom_vocab mode is active
                
                # Mode 1: Dynamic vocabulary with vocabulary table control (--custom-vocab mode)
                # This mode uses a custom vocabulary table with specific token IDs
                if "static_vocab_indices" in model_info and model_info["static_vocab_indices"] is not None:
                    static_vocab_indices = model_info["static_vocab_indices"]
                    logging.info(
                        f"Mode: Dynamic vocabulary with vocabulary table control (--custom-vocab). "
                        f"Custom vocabulary contains {len(static_vocab_indices)} token IDs"
                    )
                
                # Mode 2: Dynamic vocabulary with size control (--init-vocab-size mode)
                # This mode uses consecutive token IDs [0, 1, 2, ..., init_vocab_size-1]
                elif "init_vocab_size" in model_info:
                    init_vocab_size = int(model_info["init_vocab_size"])
                    logging.info(
                        f"Mode: Dynamic vocabulary with size control (--init-vocab-size). "
                        f"Initial vocabulary size: {init_vocab_size}"
                    )
                
                # Mode 3: Static vocabulary (no dynamic vocabulary, use full vocabulary)
                # This mode uses the complete vocabulary without dynamic management
                elif "vocab_size" in model_info:
                    init_vocab_size = int(model_info["vocab_size"])
                    use_static_vocab = model_info.get("use_static_vocab", False)
                    if use_static_vocab:
                        logging.warning(
                            "Mode: Static vocabulary detected, but use_static_vocab=True and "
                            "static_vocab_indices is missing. This might indicate a problem with "
                            "custom_vocab initialization. Falling back to full vocabulary."
                        )
                    else:
                        logging.info(
                            f"Mode: Static vocabulary (no dynamic vocabulary). "
                            f"Using full vocabulary size: {init_vocab_size}"
                        )
        except Exception as e:
            import logging
            logging.warning(f"Failed to get model info from backend: {e}")
            # Fallback to server args
            try:
                from sglang.srt.server_args import get_global_server_args
                server_args = get_global_server_args()
                # Check if custom_vocab_path is set (indicates custom_vocab mode)
                custom_vocab_path = getattr(server_args, "custom_vocab_path", None)
                if custom_vocab_path is not None:
                    logging.warning(
                        f"custom_vocab_path is set ({custom_vocab_path}) but static_vocab_indices is not available. "
                        "This might indicate a problem with custom_vocab initialization."
                    )
                if hasattr(server_args, "init_vocab_size") and server_args.init_vocab_size is not None:
                    init_vocab_size = int(server_args.init_vocab_size)
                if hasattr(server_args, "dyna_space"):
                    dyna_space = int(server_args.dyna_space)
            except (ImportError, AttributeError, ValueError):
                pass
        
        # Priority 2: Fallback to calculating from tokenizer and server args
        if static_vocab_indices is None and init_vocab_size is None:
            try:
                from sglang.srt.server_args import get_global_server_args
                server_args = get_global_server_args()
                use_static_vocab_fallback = getattr(server_args, "speculative_use_static_vocab", False)
                static_vocab_ratio = getattr(server_args, "speculative_static_vocab_ratio", 1.0)
                
                tokenizer = _ensure_tokenizer()
                full_vocab_size = getattr(tokenizer, "vocab_size", None)
                if full_vocab_size is None:
                    if hasattr(tokenizer, "vocab"):
                        full_vocab_size = len(tokenizer.vocab)
                    elif hasattr(tokenizer, "get_vocab"):
                        full_vocab_size = len(tokenizer.get_vocab())
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail="Cannot determine vocabulary size from tokenizer",
                        )
                
                if use_static_vocab_fallback:
                    init_vocab_size = max(1, int(full_vocab_size * static_vocab_ratio))
                else:
                    init_vocab_size = full_vocab_size
            except (ImportError, AttributeError):
                # Final fallback: use tokenizer vocab size
                tokenizer = _ensure_tokenizer()
                init_vocab_size = getattr(tokenizer, "vocab_size", None)
                if init_vocab_size is None:
                    if hasattr(tokenizer, "vocab"):
                        init_vocab_size = len(tokenizer.vocab)
                    elif hasattr(tokenizer, "get_vocab"):
                        init_vocab_size = len(tokenizer.get_vocab())
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail="Cannot determine vocabulary size from tokenizer or server",
                        )
        
        # Initialize vocabulary based on detected mode
        # Mode 1: Dynamic vocabulary with vocabulary table control (--custom-vocab mode)
        # Initialize with custom token IDs directly as initial vocabulary
        if static_vocab_indices is not None:
            logging.info(
                f"Initializing vocab_manager: Dynamic vocabulary with vocabulary table control. "
                f"Loading {len(static_vocab_indices)} custom token IDs"
            )
            # Get target model vocab_size for validation
            # The token_ids in static_vocab_indices should already be validated by the server,
            # but we validate again here for safety
            target_vocab_size = None
            try:
                tokenizer = _ensure_tokenizer()
                target_vocab_size = getattr(tokenizer, "vocab_size", None)
                if target_vocab_size is None:
                    if hasattr(tokenizer, "vocab"):
                        target_vocab_size = len(tokenizer.vocab)
                    elif hasattr(tokenizer, "get_vocab"):
                        target_vocab_size = len(tokenizer.get_vocab())
            except Exception:
                pass  # If we can't get vocab_size, skip validation (server already validated)
            
            vocab_manager.load_custom_vocab(
                token_ids=static_vocab_indices, 
                dyna_space=dyna_space,
                vocab_size=target_vocab_size
            )
            logging.info(
                f"vocab_manager initialized successfully: "
                f"vocab_count={len(vocab_manager.get_vocab_list())}, "
                f"initial_vocab_count={vocab_manager.get_initial_vocab_size()}"
            )
        
        # Mode 2: Dynamic vocabulary with size control (--init-vocab-size mode)
        # Initialize with consecutive token IDs [0, 1, 2, ..., init_vocab_size-1]
        elif init_vocab_size is not None:
            logging.info(
                f"Initializing vocab_manager: Dynamic vocabulary with size control. "
                f"Initial vocabulary size: {init_vocab_size}"
            )
            vocab_manager.set_dyna_space(dyna_space)
            vocab_manager.initialize_static_vocab(init_vocab_size)
            logging.info(
                f"vocab_manager initialized successfully: "
                f"vocab_count={len(vocab_manager.get_vocab_list())}, "
                f"initial_vocab_count={vocab_manager.get_initial_vocab_size()}"
            )
        
        # Mode 3: Static vocabulary (fallback - should not reach here in normal operation)
        else:
            logging.error("Failed to determine vocabulary initialization mode")
            raise HTTPException(
                status_code=500,
                detail="Cannot determine initial vocabulary configuration. "
                       "Please check server configuration and ensure either --init-vocab-size "
                       "or --custom-vocab is properly set.",
            )


def _ensure_backend():
    backend = sgl.global_config.default_backend
    if backend is None:
        backend = _get_http_backend()
        if backend is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Backend not initialized. Either call "
                    "sglang.set_default_backend(...) before launching the API "
                    f"server or set {_RUNTIME_URL_ENV} so we can proxy to an "
                    "existing runtime."
                ),
            )
        return backend
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    if not hasattr(backend, "generate"):
        raise HTTPException(status_code=500, detail="Backend missing generate()")
    return backend


def _result_meta_info(result: Any) -> Optional[Dict[str, Any]]:
    meta = getattr(result, "meta_info", None)
    if callable(meta):
        meta = meta()
    if isinstance(meta, dict):
        return meta
    return None


def _extract_avg_accept_length(meta: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract the draft model average acceptance length from meta info."""
    if not meta:
        return None
    for key in ("avg_spec_accept_length", "spec_accept_length", "accept_length"):
        value = meta.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _generate_with_hidden_states(
    backend: Any, prompt: str, sampling_kwargs: Dict[str, Any]
):
    """Invoke backend.generate ensuring hidden states are requested."""
    generate_fn = getattr(backend, "generate", None)
    if generate_fn is None:
        raise HTTPException(
            status_code=500, detail="Backend does not expose a generate() method"
        )
    try:
        return generate_fn(
            prompt,
            sampling_params=sampling_kwargs,
            return_hidden_states=True,
        )
    except TypeError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Active backend does not support return_hidden_states. "
                "Launch the runtime with --enable-return-hidden-states."
            ),
        ) from exc


# Shared helper for defaulting client IDs when omitted in a request
def _resolve_client_id(client_id: Optional[str]) -> str:
    return client_id or DEFAULT_CLIENT_ID


@app.post("/v1/vocab/add")
def add_vocab(req: VocabRequest):
    effective_client_id = _resolve_client_id(req.client_id)
    # Ensure vocabulary is initialized with initial vocab first
    _ensure_client_vocab_initialized(effective_client_id)
    tok = _ensure_tokenizer()
    token_ids = tok.convert_tokens_to_ids(req.words)
    vocab_manager.add_words(token_ids)
    vocab_count = len(vocab_manager.get_vocab_list())
    initial_vocab_count = vocab_manager.get_initial_vocab_size()
    return {
        "status": "success",
        "vocab_count": vocab_count,
        "initial_vocab_count": initial_vocab_count
    }


@app.post("/v1/vocab/remove")
def remove_vocab(req: VocabRequest):
    effective_client_id = _resolve_client_id(req.client_id)
    # Ensure vocabulary is initialized with initial vocab first
    _ensure_client_vocab_initialized(effective_client_id)
    tok = _ensure_tokenizer()
    token_ids = tok.convert_tokens_to_ids(req.words)
    vocab_manager.remove_words(token_ids)
    vocab_count = len(vocab_manager.get_vocab_list())
    initial_vocab_count = vocab_manager.get_initial_vocab_size()
    return {
        "status": "success",
        "vocab_count": vocab_count,
        "initial_vocab_count": initial_vocab_count
    }


def _build_vocab_query_response(client_id: Optional[str]) -> Dict[str, Union[str, int]]:
    effective_client_id = client_id or DEFAULT_CLIENT_ID
    _ensure_client_vocab_initialized(effective_client_id)
    vocab_count = len(vocab_manager.get_vocab_list())
    initial_vocab_count = vocab_manager.get_initial_vocab_size()
    return {
        "status": "success",
        "vocab_count": vocab_count,
        "initial_vocab_count": initial_vocab_count,
    }


@app.post("/v1/vocab/query")
def query_vocab(req: VocabQueryRequest):
    """Query vocabulary information for a client.
    
    Returns the current vocabulary count (vocab_count) and initial vocabulary count (initial_vocab_count).
    Both values are read directly from vocab_manager, which uniformly manages vocabulary data
    for both --init-vocab-size and --custom-vocab modes.
    
    - vocab_count: Current draft model vocabulary size (dynamic, changes with add/remove operations)
    - initial_vocab_count: Initial vocabulary size (static, from initial vocabulary configuration)
    If the vocabulary hasn't been initialized, it will be initialized first.
    """
    return _build_vocab_query_response(req.client_id)


@app.get("/v1/vocab/query")
def query_vocab_get(client_id: Optional[str] = None):
    """GET-friendly wrapper for querying vocab status.
    
    If client_id is omitted, a shared default client identifier is used.
    """
    return _build_vocab_query_response(client_id)


@app.post("/v1/vocab/load_custom")
def load_custom_vocab(req: CustomVocabRequest):
    """Load a custom vocabulary from a list of token IDs.
    
    This endpoint allows loading a vocabulary obtained from tokenizing datasets
    with the same model's tokenizer. The custom vocabulary will be used for
    the draft model only, while the target model uses the full vocabulary.
    
    Example:
        POST /v1/vocab/load_custom
        {
            "token_ids": [0, 1, 2, 100, 200, 300],
            "dyna_space": 1024
        }
    """
    effective_client_id = _resolve_client_id(req.client_id)
    
    # Validate token IDs
    if not req.token_ids:
        raise HTTPException(
            status_code=400,
            detail="token_ids cannot be empty"
        )
    
    # Load custom vocabulary
    vocab_manager.load_custom_vocab(
        token_ids=req.token_ids,
        dyna_space=req.dyna_space
    )
    
    vocab_count = len(vocab_manager.get_vocab_list())
    initial_vocab_count = vocab_manager.get_initial_vocab_size()
    
    return {
        "status": "success",
        "vocab_count": vocab_count,
        "initial_vocab_count": initial_vocab_count,
        "message": "Custom vocabulary loaded successfully. This vocabulary will be used for draft models only."
    }


@app.get("/get_model_info")
def get_model_info():
    """Get model information from the backend SGLang server.
    
    This endpoint proxies the request to the backend SGLang server's /get_model_info
    endpoint and returns the model information including static vocabulary settings.
    """
    try:
        backend = _ensure_backend()
        if hasattr(backend, "_get_model_info"):
            # For HttpRuntimeBackend, use _get_model_info with force_refresh
            model_info = backend._get_model_info(force_refresh=True)
            return model_info
        else:
            # Fallback: try to get from server args (only if available)
            try:
                from sglang.srt.server_args import get_global_server_args
                server_args = get_global_server_args()
                use_static_vocab = getattr(server_args, "speculative_use_static_vocab", False)
                static_vocab_ratio = getattr(server_args, "speculative_static_vocab_ratio", 1.0)
            except (ImportError, AttributeError):
                # If server_args is not available, assume static vocab is not used
                use_static_vocab = False
                static_vocab_ratio = 1.0
            
            tokenizer = _ensure_tokenizer()
            full_vocab_size = getattr(tokenizer, "vocab_size", None)
            if full_vocab_size is None:
                if hasattr(tokenizer, "vocab"):
                    full_vocab_size = len(tokenizer.vocab)
                elif hasattr(tokenizer, "get_vocab"):
                    full_vocab_size = len(tokenizer.get_vocab())
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Cannot determine vocabulary size from tokenizer"
                    )
            
            vocab_size = (
                max(1, int(full_vocab_size * static_vocab_ratio))
                if use_static_vocab
                else full_vocab_size
            )
            return {
                "vocab_size": vocab_size,
                "use_static_vocab": use_static_vocab,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/v1/generate")
def generate(req: GenerateRequest):
    effective_client_id = _resolve_client_id(req.client_id)
    # Ensure client vocabulary is initialized with static vocab first
    _ensure_client_vocab_initialized(effective_client_id)
    active_vocab_ids = vocab_manager.get_vocab_list()
    if not active_vocab_ids:
        raise HTTPException(
            status_code=400,
            detail=f"No active vocabulary for client {effective_client_id}",
        )

    sampling_kwargs = dict(req.sampling_params or {})
    sampling_kwargs["dynamic_vocab_token_ids"] = active_vocab_ids
    if req.max_new_tokens is not None and "max_new_tokens" not in sampling_kwargs:
        sampling_kwargs["max_new_tokens"] = req.max_new_tokens
    if req.temperature is not None and "temperature" not in sampling_kwargs:
        sampling_kwargs["temperature"] = req.temperature

    backend = _ensure_backend()
    result = backend.generate(
        req.prompt,
        sampling_params=sampling_kwargs,
    )
    response = {
        "text": result.text(),
    }
    meta = _result_meta_info(result)
    avg_accept_length = _extract_avg_accept_length(meta)
    if avg_accept_length is not None:
        response["draft_avg_accept_length"] = avg_accept_length
    # if meta is not None:
    #     response["meta_info"] = meta
    return response


@app.post("/v1/hidden_states")
def get_hidden_states(req: HiddenStateRequest):
    """Fetch hidden states (last-layer features) for a prompt."""
    effective_client_id = _resolve_client_id(req.client_id)
    _ensure_client_vocab_initialized(effective_client_id)
    active_vocab_ids = vocab_manager.get_vocab_list()
    if not active_vocab_ids:
        raise HTTPException(
            status_code=400,
            detail=f"No active vocabulary for client {effective_client_id}",
        )

    sampling_kwargs = dict(req.sampling_params or {})
    sampling_kwargs["dynamic_vocab_token_ids"] = active_vocab_ids

    if req.max_new_tokens is not None and "max_new_tokens" not in sampling_kwargs:
        sampling_kwargs["max_new_tokens"] = req.max_new_tokens
    else:
        sampling_kwargs.setdefault("max_new_tokens", 0)

    if req.temperature is not None and "temperature" not in sampling_kwargs:
        sampling_kwargs["temperature"] = req.temperature

    backend = _ensure_backend()
    result = _generate_with_hidden_states(backend, req.prompt, sampling_kwargs)
    meta = _result_meta_info(result) or {}
    hidden_states = meta.get("hidden_states")
    if hidden_states is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Backend did not return hidden states. "
                "Make sure the runtime was launched with --enable-return-hidden-states."
            ),
        )

    return {
        "hidden_states": hidden_states,
        "meta_info": meta,
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if req.stream:
        raise HTTPException(
            status_code=501, detail="Streaming is not yet supported"
        )
    
    # Get tokenizer
    tokenizer = _ensure_tokenizer()
    
    # Convert messages to prompt using chat template
    try:
        # Convert messages to format expected by apply_chat_template
        messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        # Ensure prompt_ids is a list
        if not isinstance(prompt_ids, list):
            prompt_ids = tokenizer.encode(prompt_ids)
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process messages: {str(e)}"
        )
    
    # Prepare sampling parameters
    sampling_kwargs = {}
    if req.max_tokens is not None:
        sampling_kwargs["max_new_tokens"] = req.max_tokens
    if req.temperature is not None:
        sampling_kwargs["temperature"] = req.temperature
    
    # Add dynamic vocab if client_id is provided
    if req.client_id:
        # Ensure client vocabulary is initialized with static vocab first
        _ensure_client_vocab_initialized(req.client_id)
        active_vocab_ids = vocab_manager.get_vocab_list()
        if active_vocab_ids:
            sampling_kwargs["dynamic_vocab_token_ids"] = active_vocab_ids
    
    # Generate response
    backend = _ensure_backend()
    result = backend.generate(
        prompt,
        sampling_params=sampling_kwargs,
    )
    
    generated_text = result.text()
    
    # Calculate token usage
    prompt_tokens = len(prompt_ids)
    completion_tokens = len(tokenizer.encode(generated_text))
    
    # Build response
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    response = ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    
    return response


def _get_http_backend() -> Optional["_HttpRuntimeBackend"]:
    global _http_backend
    runtime_url = os.environ.get(_RUNTIME_URL_ENV)
    if not runtime_url:
        return None

    with _http_backend_lock:
        if _http_backend is not None:
            return _http_backend

        api_key = os.environ.get(_RUNTIME_API_KEY_ENV)
        verify = _coerce_verify(os.environ.get(_RUNTIME_VERIFY_ENV))
        timeout = _coerce_timeout(os.environ.get(_RUNTIME_TIMEOUT_ENV))
        _http_backend = _HttpRuntimeBackend(
            runtime_url.rstrip("/"), api_key=api_key, verify=verify, timeout=timeout
        )
        return _http_backend


def _coerce_verify(raw: Optional[str]) -> Union[bool, str]:
    if raw is None or raw == "":
        return True
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    return raw


def _coerce_timeout(raw: Optional[str]) -> float:
    if not raw:
        return _DEFAULT_HTTP_TIMEOUT
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_HTTP_TIMEOUT


class _HttpGenerateResult:
    def __init__(self, text: str, meta_info: Optional[Dict[str, Any]] = None):
        self._text = text
        self._meta_info = meta_info or {}

    def text(self) -> str:
        return self._text

    def meta_info(self) -> Dict[str, Any]:
        return self._meta_info


class _HttpRuntimeBackend:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str],
        verify: Union[bool, str],
        timeout: float,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.verify = verify
        self.timeout = timeout
        self._session = requests.Session()
        self._model_info: Optional[dict] = None
        self._tokenizer = None
        self._tokenizer_lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        sampling_params: dict,
        *,
        return_hidden_states: bool = False,
    ):
        response = self._request(
            "POST",
            "/generate",
            json={
                "text": prompt,
                "sampling_params": sampling_params,
                **({"return_hidden_states": True} if return_hidden_states else {}),
            },
        )
        payload = response.json()
        text = _extract_text(payload)
        if text is None:
            raise RuntimeError("Runtime response did not include 'text'")
        meta_info = _extract_meta_info(payload)
        return _HttpGenerateResult(text, meta_info)

    def get_tokenizer(self):
        with self._tokenizer_lock:
            if self._tokenizer is None:
                tokenizer_source = (
                    os.environ.get(_TOKENIZER_OVERRIDE_ENV)
                    or self._get_model_info().get("tokenizer_path")
                    or self._get_model_info().get("model_path")
                )
                if not tokenizer_source:
                    raise RuntimeError(
                        "Unable to determine tokenizer path from runtime metadata. "
                        f"Set {_TOKENIZER_OVERRIDE_ENV} to override."
                    )
                from sglang.srt.utils.hf_transformers_utils import get_tokenizer

                self._tokenizer = get_tokenizer(tokenizer_source)
            return self._tokenizer

    def _get_model_info(self, force_refresh: bool = False) -> dict:
        if self._model_info is None or force_refresh:
            response = self._request("GET", "/get_model_info")
            try:
                self._model_info = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    "Failed to parse /get_model_info response from runtime"
                ) from exc
        return self._model_info

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key}", **headers}
        try:
            response = self._session.request(
                method,
                url,
                headers=headers,
                timeout=kwargs.pop("timeout", self.timeout),
                verify=self.verify,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            snippet = ""
            if hasattr(exc, "response") and exc.response is not None:
                snippet = exc.response.text[:256].strip()
            raise RuntimeError(
                f"Request to runtime {url} failed: {exc}. {snippet}"
            ) from exc


def _extract_text(payload) -> Optional[str]:
    if isinstance(payload, dict):
        text_field = payload.get("text")
        if isinstance(text_field, list):
            return text_field[0] if text_field else ""
        return text_field

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            text_field = item.get("text")
            if text_field is None:
                continue
            if isinstance(text_field, list):
                return text_field[0] if text_field else ""
            return text_field

    return None


def _extract_meta_info(payload) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        meta = payload.get("meta_info")
        if isinstance(meta, dict):
            return meta

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            meta = item.get("meta_info")
            if isinstance(meta, dict):
                return meta

    return None
