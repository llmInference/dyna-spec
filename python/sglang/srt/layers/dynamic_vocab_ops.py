from typing import Union

import torch
import triton
import triton.language as tl


@triton.jit
def dynamic_vocab_matmul_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    M,
    K,
    Capacity,
    PopulatedSize_ptr,  # Changed to ptr
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load populated_size from pointer (scalar load)
    # Using a pointer allows update without graph recapture
    PopulatedSize = tl.load(PopulatedSize_ptr)

    # Range of output rows (M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Range of output cols (N/Capacity)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Early exit for blocks completely outside populated size
    # Note: We must still write something (e.g. -inf) if we want valid output,
    # or we assume output is pre-filled or we don't care about garbage in unused slots.
    # But to be safe and clean, we can write -inf.
    # However, for "saving compute", we want to skip the loop.

    # Check if this block overlaps with populated region
    block_start_n = pid_n * BLOCK_N
    if block_start_n >= PopulatedSize:
        # Completely outside. Write -inf and exit.
        # We need to mask valid M.
        mask_m = offs_m < M
        mask_n = offs_n < Capacity

        # Create pointers
        out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)

        # Write -inf
        tl.store(out_ptrs, float("-inf"), mask=mask_m[:, None] & mask_n[None, :])
        return

    # Pointers for X and W
    # X: [M, K]
    # W: [Capacity, K] (Assuming W is stored as [Capacity, K] row-major)
    # We want Out = X @ W.T

    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load X
        # Mask for X: M is bounded, K is bounded
        mask_x = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Load W
        # Mask for W: N is bounded by Capacity (and PopulatedSize effectively), K is bounded
        # We load up to Capacity.
        mask_w = (offs_n[None, :] < Capacity) & (k + offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # Matmul
        accumulator += tl.dot(x, w)

        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Store output
    # Mask for output
    mask_m = offs_m < M
    mask_n = offs_n < Capacity

    # Apply mask for populated size to write -inf for unpopulated slots within this block
    # (The block might partially overlap populated region)
    mask_populated = offs_n < PopulatedSize

    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)

    # We write result where populated, -inf where not populated
    # But accumulator has values.
    # We can use `tl.where`

    final_val = tl.where(mask_populated[None, :], accumulator, float("-inf"))

    tl.store(out_ptrs, final_val.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def dynamic_vocab_matmul(
    x: torch.Tensor, w: torch.Tensor, populated_size: Union[int, torch.Tensor]
):
    """
    x: [M, K]
    w: [Capacity, K]
    populated_size: int or Tensor (scalar)
    Returns: [M, Capacity]
    """
    M, K = x.shape
    Capacity, K_w = w.shape
    assert K == K_w

    out = torch.empty((M, Capacity), device=x.device, dtype=torch.float16)

    # Ensure populated_size is provided as a pointer for the kernel if it's a tensor
    # If it's an int, we should probably wrap it in a tensor to enable graph capture robustness?
    # But ideally callers should pass a tensor.
    # For now, if int, we create a temporary tensor (but this won't help with updates
    # unless the CUDA Graph captures the pointer to a PERSISTENT tensor).
    # So we assume the caller provides a persistent tensor for Graph usage.

    populated_size_ptr = populated_size
    if isinstance(populated_size, int):
        # Warning: If capturing graph with this, it will be baked as a constant
        # because this tensor is temporary.
        # But for eager mode, it's fine.
        populated_size_ptr = torch.tensor(
            [populated_size], dtype=torch.int32, device=x.device
        )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(Capacity, META["BLOCK_N"]),
    )

    dynamic_vocab_matmul_kernel[grid](
        x,
        w,
        out,
        M,
        K,
        Capacity,
        populated_size_ptr,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_K=32,
    )

    return out
