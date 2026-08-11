# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward-only Triton kernels for Motif MHC layers.

Vendored standalone from `llm_training/layers/triton_{sinkhorn,res,mhc_post}_kernel.py`
(Motif Technologies). Backward / autograd / scratchpad paths are stripped since
vLLM is forward-only.

Public API:
    sinkhorn_fused(base_matrix: fp32 [..., R, R], K) -> fp32 [..., R, R]
        K-iteration Sinkhorn-Knopp doubly-stochastic normalization, all
        iterations folded into one kernel launch.

    res_triton(h_res [B,S,R,R], x [B,S,R,D]) -> [B,S,R,D]
        Batched small matmul; equivalent to einsum("bsij,bsjd->bsid", h_res, x).

    mhc_post_fused(h_res [B,S,R,R], x [B,S,R,D],
                   h_post [B,S,R], sublayer_out [B,S,D]) -> [B,S,R,D]
        Fused: h_res @ x + h_post[..., None] * sublayer_out[..., None, :].
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# =============================================================================
# Sinkhorn-Knopp
# =============================================================================

@triton.jit
def _sinkhorn_fwd_kernel(
    base_ptr,
    out_ptr,
    BS,
    stride_pos,
    K: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    base = base_ptr + pid * stride_pos
    out = out_ptr + pid * stride_pos

    row_idx = tl.arange(0, BLOCK_R)[:, None]
    col_idx = tl.arange(0, BLOCK_R)[None, :]
    offsets = row_idx * R + col_idx
    mask = (row_idx < R) & (col_idx < R)

    b = tl.load(base + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.minimum(tl.maximum(b, -20.0), 20.0)
    m = tl.exp(b)

    for _k in range(K):
        row_sum = tl.sum(m, axis=1)[:, None]
        row_sum = tl.maximum(row_sum, 1e-8)
        m = m / row_sum
        col_sum = tl.sum(m, axis=0)[None, :]
        col_sum = tl.maximum(col_sum, 1e-8)
        m = m / col_sum

    tl.store(out + offsets, m, mask=mask)


@torch.library.custom_op("motif::sinkhorn_fused", mutates_args=())
def sinkhorn_fused(base_matrix: torch.Tensor, K: int) -> torch.Tensor:
    """K-iteration Sinkhorn-Knopp on (..., R, R) fp32 matrices."""
    assert base_matrix.dtype == torch.float32, (
        f"base_matrix must be fp32, got {base_matrix.dtype}"
    )
    R = base_matrix.shape[-1]
    assert base_matrix.shape[-2] == R, "Last two dims must be RxR"

    base_matrix = base_matrix.contiguous()
    out = torch.empty_like(base_matrix)
    BS = base_matrix.numel() // (R * R)
    if BS == 0:
        return out
    BLOCK_R = _next_power_of_2(R)

    _sinkhorn_fwd_kernel[(BS,)](
        base_matrix,
        out,
        BS,
        stride_pos=R * R,
        K=K,
        R=R,
        BLOCK_R=BLOCK_R,
    )
    return out


@sinkhorn_fused.register_fake
def _sinkhorn_fused_fake(base_matrix: torch.Tensor, K: int) -> torch.Tensor:
    return torch.empty_like(base_matrix)


# =============================================================================
# res_matmul: einsum("bsij,bsjd->bsid", h_res, x)
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}),
        triton.Config({"BLOCK_D": 128}),
        triton.Config({"BLOCK_D": 256}),
    ],
    key=["D"],
)
@triton.jit
def _res_matmul_fwd_kernel(
    h_ptr,
    x_ptr,
    out_ptr,
    BS,
    D,
    D_PADDED: tl.constexpr,
    stride_h_pos,
    stride_h_row,
    stride_x_pos,
    stride_x_row,
    stride_o_pos,
    stride_o_row,
    R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h_base = h_ptr + pid * stride_h_pos
    x_base = x_ptr + pid * stride_x_pos
    o_base = out_ptr + pid * stride_o_pos

    for d_start in tl.static_range(0, D_PADDED, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        for i in range(R):
            acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for j in range(R):
                h_ij = tl.load(h_base + i * stride_h_row + j).to(tl.float32)
                x_j = tl.load(
                    x_base + j * stride_x_row + d_offs,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                acc += h_ij * x_j
            tl.store(o_base + i * stride_o_row + d_offs, acc, mask=mask)


@torch.library.custom_op("motif::res_triton", mutates_args=())
def res_triton(h_res: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Equivalent to einsum("bsij,bsjd->bsid", h_res, x)."""
    B, S, R, D = x.shape
    assert h_res.shape == (B, S, R, R), (
        f"h_res shape mismatch: expected {(B, S, R, R)}, got {h_res.shape}"
    )

    h_res = h_res.contiguous()
    x = x.contiguous()
    out = torch.empty_like(x)

    BS = B * S
    if BS == 0:
        return out

    BLOCK_D_MAX = 256
    D_PADDED = ((D + BLOCK_D_MAX - 1) // BLOCK_D_MAX) * BLOCK_D_MAX

    _res_matmul_fwd_kernel[(BS,)](
        h_res,
        x,
        out,
        BS,
        D,
        D_PADDED=D_PADDED,
        stride_h_pos=R * R,
        stride_h_row=R,
        stride_x_pos=R * D,
        stride_x_row=D,
        stride_o_pos=R * D,
        stride_o_row=D,
        R=R,
    )
    return out


@res_triton.register_fake
def _res_triton_fake(h_res: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


# =============================================================================
# mhc_post_fused: out[i,d] = sum_j h_res[i,j] * x[j,d] + h_post[i] * sub[d]
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}),
        triton.Config({"BLOCK_D": 128}),
        triton.Config({"BLOCK_D": 256}),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_fwd_kernel(
    h_res_ptr,
    x_ptr,
    h_post_ptr,
    sub_ptr,
    out_ptr,
    BS,
    D,
    D_PADDED: tl.constexpr,
    stride_hr_pos,
    stride_hr_row,
    stride_x_pos,
    stride_x_row,
    stride_hp_pos,
    stride_sub_pos,
    stride_o_pos,
    stride_o_row,
    R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    hr_base = h_res_ptr + pid * stride_hr_pos
    x_base = x_ptr + pid * stride_x_pos
    hp_base = h_post_ptr + pid * stride_hp_pos
    sub_base = sub_ptr + pid * stride_sub_pos
    o_base = out_ptr + pid * stride_o_pos

    for d_start in tl.static_range(0, D_PADDED, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        sub_d = tl.load(sub_base + d_offs, mask=mask, other=0.0).to(tl.float32)

        for i in range(R):
            hp_i = tl.load(hp_base + i).to(tl.float32)
            acc = hp_i * sub_d
            for j in range(R):
                hr_ij = tl.load(hr_base + i * stride_hr_row + j).to(tl.float32)
                x_jd = tl.load(
                    x_base + j * stride_x_row + d_offs,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                acc += hr_ij * x_jd
            tl.store(o_base + i * stride_o_row + d_offs, acc, mask=mask)


@torch.library.custom_op("motif::mhc_post_fused", mutates_args=())
def mhc_post_fused(
    h_res: torch.Tensor,
    x: torch.Tensor,
    h_post: torch.Tensor,
    sublayer_out: torch.Tensor,
) -> torch.Tensor:
    """Fused: h_res @ x + h_post[..., None] * sublayer_out[..., None, :].

    Shapes:
        h_res        : (B, S, R, R)
        x            : (B, S, R, D)
        h_post       : (B, S, R)
        sublayer_out : (B, S, D)
        returns      : (B, S, R, D)
    """
    B, S, R, D = x.shape
    h_res = h_res.contiguous()
    x = x.contiguous()
    h_post = h_post.contiguous()
    sublayer_out = sublayer_out.contiguous()
    out = torch.empty_like(x)

    BS = B * S
    if BS == 0:
        return out

    BLOCK_D_MAX = 256
    D_PADDED = ((D + BLOCK_D_MAX - 1) // BLOCK_D_MAX) * BLOCK_D_MAX

    _mhc_post_fwd_kernel[(BS,)](
        h_res,
        x,
        h_post,
        sublayer_out,
        out,
        BS,
        D,
        D_PADDED=D_PADDED,
        stride_hr_pos=R * R,
        stride_hr_row=R,
        stride_x_pos=R * D,
        stride_x_row=D,
        stride_hp_pos=R,
        stride_sub_pos=D,
        stride_o_pos=R * D,
        stride_o_row=D,
        R=R,
    )
    return out


@mhc_post_fused.register_fake
def _mhc_post_fused_fake(
    h_res: torch.Tensor,
    x: torch.Tensor,
    h_post: torch.Tensor,
    sublayer_out: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)
