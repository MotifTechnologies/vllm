# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motif MoE expert kernel for MXFP8 (auto bf16->MXFP8 at load time)."""

from __future__ import annotations

import torch
import torch.nn as nn

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.motif_experts import (
    _MotifPolyNormExpertsBase,
    _MotifPolyNormMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


def _quantize_experts_to_mxfp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE weight ``[E, X, K]`` (bf16) to MXFP8.

    Returns the B-operand layout for ``cutlass_mxfp8_grouped_mm``:
    ``[E, K, X]`` fp8 + ``[E, K_blocks, X]`` uint8 scales (K-contiguous, i.e.
    column-major; ``.contiguous()`` would break the kernel's layout check).
    """
    assert weight.dim() == 3, f"weight must be 3D, got {weight.shape}"
    E, X, K = weight.shape
    assert K % MXFP8_BLOCK_SIZE == 0, f"K={K} % {MXFP8_BLOCK_SIZE} != 0"
    assert X % 128 == 0, f"X={X} must be 128-aligned for MXFP8 scale layout"
    K_blocks = K // MXFP8_BLOCK_SIZE
    device = weight.device

    problem_sizes = torch.tensor(
        [[X, K, K]] * E, dtype=torch.int32, device=device
    )
    expert_offsets = torch.tensor(
        [i * X for i in range(E)], dtype=torch.int32, device=device
    )
    blockscale_offsets = expert_offsets.clone()  # X already 128-aligned.

    fp_out = torch.empty(E * X, K, dtype=MXFP8_VALUE_DTYPE, device=device)
    scale_out = torch.empty(
        E * X, K_blocks, dtype=MXFP8_SCALE_DTYPE, device=device
    )
    ops.mxfp8_experts_quant(
        weight.reshape(E * X, K).contiguous(),
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        fp_out,
        scale_out,
    )

    weight_fp8 = fp_out.view(E, X, K).transpose(1, 2)
    weight_scale = scale_out.view(E, X, K_blocks).transpose(1, 2)
    return weight_fp8, weight_scale


class MotifMxfp8Experts(_MotifPolyNormExpertsBase):
    """Motif MoE experts using CUTLASS MXFP8 grouped MM.

    Per-expert weight layout (after process_weights_after_loading):
      ``w1`` (gate||up): ``[E_local, K, 2*I]`` fp8 + ``[E_local, K/32, 2*I]`` uint8
      ``w2`` (down):     ``[E_local, I, K]`` fp8 + ``[E_local, I/32, K]`` uint8

    Flow mirrors ``run_cutlass_moe_fp8``: moe_permute → quantize input →
    cutlass_mxfp8_grouped_mm (GEMM1) → grouped_poly_norm_forward → quantize
    again → cutlass_mxfp8_grouped_mm (GEMM2) → moe_unpermute (folds topk
    weights + reduce).
    """

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.has_device_capability(100)
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Mirrors CutlassExpertsFp8.workspace_shapes. N=2*I (gated).
        activation_out_dim = N // 2
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(activation_out_dim, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert hidden_states.is_contiguous()
        assert hidden_states.dim() == 2
        assert hidden_states.dtype in (torch.float16, torch.bfloat16)
        assert not apply_router_weight_on_input, (
            "MotifMxfp8Experts does not support apply_router_weight_on_input"
        )

        E_local, K, two_intermediate_size = w1.shape
        intermediate_size = two_intermediate_size // 2
        device = hidden_states.device
        M = hidden_states.size(0)
        topk = topk_ids.size(1)
        M_total = M * topk

        if global_num_experts == -1:
            global_num_experts = E_local

        # a_perm uses a fresh buffer (workspace2 is reserved for gemm2_out;
        # see cutlass_moe.py: mm2_out must NOT alias output, which shares
        # memory with workspace13 in the non-chunked path).
        permuted_buf = torch.empty(
            (M_total, K), dtype=hidden_states.dtype, device=device
        )
        a_perm, _, expert_first_token_offset, inv_permuted_idx, _ = moe_permute(
            hidden_states,
            a1q_scale=None,
            topk_ids=topk_ids,
            n_expert=global_num_experts,
            n_local_expert=E_local,
            expert_map=expert_map,
            permuted_hidden_states=permuted_buf,
        )

        problem_sizes1 = torch.empty(
            (E_local, 3), dtype=torch.int32, device=device
        )
        problem_sizes2 = torch.empty(
            (E_local, 3), dtype=torch.int32, device=device
        )
        ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
            expert_first_token_offset, problem_sizes1, problem_sizes2,
            intermediate_size, K, False,
        )
        expert_offsets = expert_first_token_offset[:-1].to(torch.int32)
        # Per-expert MXFP8 scale slice is align(m_g, 128) rows; offsets are
        # the cumulative aligned counts. Use a Python-side upper bound for
        # the scale buffer to avoid a CPU sync on the cumsum (forbidden
        # during CUDA graph capture).
        counts = (
            expert_first_token_offset[1:] - expert_first_token_offset[:-1]
        ).to(torch.int32)
        aligned = ((counts + 127) // 128) * 128
        blockscale_offsets = torch.zeros(
            (E_local,), dtype=torch.int32, device=device
        )
        blockscale_offsets[1:] = aligned[:-1].cumsum(0).to(torch.int32)
        scale_buf_rows = M_total + E_local * 128

        K_blocks = K // MXFP8_BLOCK_SIZE
        a_perm_fp8 = torch.empty(
            (M_total, K), dtype=MXFP8_VALUE_DTYPE, device=device
        )
        a_perm_scale = torch.empty(
            (scale_buf_rows, K_blocks), dtype=MXFP8_SCALE_DTYPE, device=device,
        )
        ops.mxfp8_experts_quant(
            a_perm, problem_sizes1, expert_offsets, blockscale_offsets,
            a_perm_fp8, a_perm_scale,
        )

        gemm1_out = _resize_cache(workspace13, (M_total, two_intermediate_size))
        ops.cutlass_mxfp8_grouped_mm(
            a_perm_fp8, w1, a_perm_scale, self.w1_scale, gemm1_out,
            problem_sizes1, expert_offsets, blockscale_offsets,
        )

        # PolyNorm operates on the permuted layout. We hand it a fake
        # top_k=1 with a per-row local expert id derived from the cumulative
        # expert_first_token_offset (searchsorted instead of repeat_interleave
        # to stay CPU-sync-free during CUDA graph capture).
        all_rows = torch.arange(M_total, device=device, dtype=torch.int64)
        permuted_topk_ids = (
            torch.searchsorted(
                expert_first_token_offset, all_rows, right=True
            ) - 1
        ).clamp_(0, E_local - 1).to(torch.int32).unsqueeze(1)

        gate, up = gemm1_out.split(intermediate_size, dim=-1)
        act_out = self._grouped_polynorm_activation(
            gate.contiguous(), up.contiguous(), permuted_topk_ids, None, 1
        )

        intermediate_blocks = intermediate_size // MXFP8_BLOCK_SIZE
        act_fp8 = torch.empty(
            (M_total, intermediate_size), dtype=MXFP8_VALUE_DTYPE, device=device
        )
        act_scale = torch.empty(
            (scale_buf_rows, intermediate_blocks),
            dtype=MXFP8_SCALE_DTYPE, device=device,
        )
        ops.mxfp8_experts_quant(
            act_out.contiguous(), problem_sizes2,
            expert_offsets, blockscale_offsets, act_fp8, act_scale,
        )

        # gemm2_out MUST use workspace2 (not workspace13) — workspace13
        # aliases ``output`` in the non-chunked path and moe_unpermute would
        # then read its own writes.
        gemm2_out = _resize_cache(workspace2, (M_total, K))
        ops.cutlass_mxfp8_grouped_mm(
            act_fp8, w2, act_scale, self.w2_scale, gemm2_out,
            problem_sizes2, expert_offsets, blockscale_offsets,
        )

        # moe_unpermute folds topk_weights + reduces over the topk dim.
        # CHECK_SKIPPED branch handles the EP sentinel in inv_permuted_idx.
        moe_unpermute(
            out=output,
            permuted_hidden_states=gemm2_out,
            topk_weights=topk_weights,
            inv_permuted_idx=inv_permuted_idx,
            expert_first_token_offset=expert_first_token_offset,
        )


class MotifMxfp8MoEMethod(_MotifPolyNormMoEMethodBase):
    """MoE method that auto-converts bf16 weights to MXFP8 at load time.

    Inherits ``UnquantizedFusedMoEMethod`` (via ``_MotifPolyNormMoEMethodBase``,
    a ``nn.Module``) so it can be installed via
    ``FusedMoE._replace_quant_method``. ``SharedFusedMoE`` is constructed with
    ``quant_config=None`` so the upstream method allocates bf16 weights from the
    checkpoint; ``process_weights_after_loading`` then in-place quantizes them
    to MXFP8 (fp8 + E8M0 uint8 scales).

    ``super().process_weights_after_loading`` is intentionally skipped — it
    runs the unquantized kernel's ``_setup_kernel`` which under
    ``VLLM_USE_FLASHINFER_MOE_FP16=1`` swaps w13->w31 and breaks gate/up
    split.
    """

    @property
    def supports_eplb(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if not (
            current_platform.is_cuda()
            and current_platform.has_device_capability(100)
        ):
            raise RuntimeError(
                "MotifMxfp8MoEMethod requires SM100 (Blackwell) or newer."
            )

        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        assert w13.dtype in (torch.bfloat16, torch.float16), (
            f"expected bf16/fp16 w13 at load time, got {w13.dtype}"
        )
        w13_fp8, w13_scale = _quantize_experts_to_mxfp8(w13)
        w2_fp8, w2_scale = _quantize_experts_to_mxfp8(w2)

        replace_parameter(layer, "w13_weight", w13_fp8)
        replace_parameter(layer, "w2_weight", w2_fp8)
        layer.register_parameter(
            "w13_weight_scale", nn.Parameter(w13_scale, requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_scale", nn.Parameter(w2_scale, requires_grad=False),
        )
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def get_fused_moe_quant_config(
        self, layer: nn.Module
    ) -> FusedMoEQuantConfig | None:
        # _a1/_a2 dtype=None: the prepare/dispatch flow ships bf16 tokens
        # untouched. If we advertised fp8_e4m3fn here,
        # _quantize_and_setup_dispatch would call _fp8_quantize (plain
        # per-tensor fp8, NOT MXFP8) and drop the scale before apply().
        a_desc = FusedMoEQuantDesc()
        block_shape = GroupShape(1, MXFP8_BLOCK_SIZE)
        w1_desc = FusedMoEQuantDesc(
            dtype=MXFP8_VALUE_DTYPE, shape=block_shape,
            scale=layer.w13_weight_scale,
        )
        w2_desc = FusedMoEQuantDesc(
            dtype=MXFP8_VALUE_DTYPE, shape=block_shape,
            scale=layer.w2_weight_scale,
        )
        return FusedMoEQuantConfig(
            _a1=a_desc, _a2=a_desc, _w1=w1_desc, _w2=w2_desc,
            is_nvfp4_scale_swizzled=False,
        )

    def select_gemm_impl(self, prepare_finalize, layer):
        assert (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.Standard
        )
        assert self.moe_quant_config is not None
        return MotifMxfp8Experts(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            poly_norm_weight=self.poly_norm_weight,
            poly_norm_bias=self.poly_norm_bias,
            hidden_clamp=self.hidden_clamp,
            polynorm_output_scale=self.polynorm_output_scale,
            polynorm_sigmoid_weight=self.polynorm_sigmoid_weight,
        )

    def apply(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "MotifMxfp8MoEMethod.apply should not be called; the modular "
            "kernel dispatches through MotifMxfp8Experts."
        )
