# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motif MoE experts via DeepGEMM 1x128 grouped GEMM + grouped PolyNorm.

This mirrors the stock contiguous DeepGEMM MoE flow exactly
(``deepgemm_moe_permute`` -> GEMM1 -> activation -> GEMM2 ->
``deepgemm_unpermute_and_reduce``); the ONLY change is the inter-GEMM
activation: motif's per-expert grouped PolyNorm replaces SiLU+mul.

1x128 (standard DeepSeek 1d2d recipe: act per-token-group 1x128, weight
128x128) is used first because vLLM's DeepGEMM permute/scatter machinery is
1x128-native. A later 1x32 (MX) upgrade only needs the permute wrapper to carry
1x32 scales; the apply() structure here is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
    expert_aligned_psum,
    get_mk_alignment_for_contiguous_layout,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.motif_experts import (
    _MotifPolyNormMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
)

# DeepGEMM 1d2d recipe: 1x128 per-token-group activation, 128x128 block weight.
_FP8_DTYPE = torch.float8_e4m3fn
_BLOCK = 128


def _block_quant_experts_to_fp8(
    weight: torch.Tensor, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE weight ``[E, N, K]`` (bf16) to fp8 128x128 blocks.

    Returns ``[E, N, K]`` fp8 + ``[E, ceil(N/128), ceil(K/128)]`` fp32 scales,
    the layout DeepGEMM's ``m_grouped_fp8_gemm_nt_contiguous`` consumes for the
    B operand. Uses vLLM's ``per_block_cast_to_fp8`` wrapper (same scale layout
    as the kernel), which falls back to ``vllm.third_party.deep_gemm`` when the
    standalone ``deep_gemm`` package is not installed.
    """
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    assert weight.dim() == 3, f"weight must be 3D, got {weight.shape}"
    fp8_experts, scale_experts = [], []
    for e in range(weight.size(0)):
        fp8_e, sf_e = per_block_cast_to_fp8(
            weight[e], block_size=[_BLOCK, _BLOCK], use_ue8m0=use_ue8m0
        )
        fp8_experts.append(fp8_e)
        scale_experts.append(sf_e)
    return torch.stack(fp8_experts), torch.stack(scale_experts)


class MotifDeepGemmExperts(DeepGemmExperts):
    """DeepGEMM (fp8 1x128) experts with motif's grouped PolyNorm activation.

    Per-expert PolyNorm parameters (``poly_norm_weight [E_local, 3]``,
    ``poly_norm_bias [E_local, 1]``) are owned by the caller and passed in. The
    count-persistent CUDA kernel ``grouped_poly_norm_fp8_quant`` fuses gate/up
    clamp + polynom + mul + output clamp/scale + 1x128 fp8 requant in one launch,
    processing only valid rows per expert (padding skipped, matching the
    psum-layout GEMM).
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        poly_norm_weight: torch.Tensor,
        poly_norm_bias: torch.Tensor,
        hidden_clamp: float | None,
        polynorm_output_scale: float,
        polynorm_sigmoid_weight: bool,
        eps: float = 1e-6,
    ):
        # Skip DeepGemmExperts.__init__'s asserts (quant_dtype == fp8e4m3 and
        # block_shape == alignment): motif ships bf16 and quantizes inside
        # apply(), so the activation desc carries no dtype (quant_dtype=None).
        # Initialize the grandparent (FusedMoEExpertsModular) directly.
        mk.FusedMoEExpertsModular.__init__(
            self, moe_config=moe_config, quant_config=quant_config
        )
        self.poly_norm_weight = poly_norm_weight
        self.poly_norm_bias = poly_norm_bias
        self.hidden_clamp = hidden_clamp
        self.polynorm_output_scale = polynorm_output_scale
        self.polynorm_sigmoid_weight = polynorm_sigmoid_weight
        self.eps = eps
        # Lazy fp32 (+sigmoid) cache of the PolyNorm params — built on first
        # apply() (post weight-load); saves a cast+sigmoid launch per MoE
        # layer per step. Weights are static during serving; an RL refit
        # (in-place weight push) must call refresh_pn_params() afterwards —
        # NeMo-RL does this from its post-refit hook, mirroring the tilelang
        # MHC graft.
        self._pn_w_f32: torch.Tensor | None = None
        self._pn_b_f32: torch.Tensor | None = None

    def _compute_pn_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.poly_norm_weight.float()
        if self.polynorm_sigmoid_weight:
            w = torch.sigmoid(w)
        return w, self.poly_norm_bias.float()

    def _pn_params_f32(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pn_w_f32 is None:
            self._pn_w_f32, self._pn_b_f32 = self._compute_pn_params()
        assert self._pn_b_f32 is not None
        return self._pn_w_f32, self._pn_b_f32

    def refresh_pn_params(self) -> None:
        """RL-refit hook: recompute the cached fold IN PLACE (cudagraphs
        capture the cached tensors' addresses). No-op if never built — the
        lazy path then builds fresh from the refit weights."""
        if self._pn_w_f32 is not None:
            wf, bf = self._compute_pn_params()
            self._pn_w_f32.copy_(wf)
            assert self._pn_b_f32 is not None
            self._pn_b_f32.copy_(bf)

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
        # Mirrors DeepGemmExperts.apply; differences vs stock:
        #  (1) input is quantized HERE (bf16 ships through prepare untouched,
        #      like MotifMxfp8Experts) — quantize per-token ONCE before the
        #      permute replicates x across top_k, then permute the fp8;
        #  (2) the inter-GEMM activation is grouped PolyNorm, not SiLU+mul.
        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None
        assert hidden_states.dtype in (
            torch.float16, torch.bfloat16, torch.float8_e4m3fn
        )
        # Packed-UE8M0 SF layout is an SM100 (1d1d) format; SM90 (1d2d)
        # kernels assert fp32 SF dtypes.
        _sf_packed = current_platform.has_device_capability(100)

        # Per-token 1x128 quant (row-major scales: deepgemm_moe_permute's
        # ep_scatter reads recv_x_scale[token, :] row-wise). bf16 ships through
        # prepare untouched (like MotifMxfp8Experts); quantize the input here.
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
        if hidden_states.dtype == torch.float8_e4m3fn:
            # Prepare already quantized (fp8 shipped through the EP allgather
            # at half the bytes of bf16); reuse its 1x128 row-major scales.
            assert a1q_scale is not None
            a1q = hidden_states
            a1q_scale = a1q_scale.contiguous()
        else:
            a1q, a1q_scale = per_token_group_quant_fp8(
                hidden_states, self.block_shape[1], column_major_scales=False,
            )
        _, N, K = w1.size()  # w1: [E_local, 2*I, K]
        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts
        assert w2.size(1) == K

        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        # expert_ids (m_indices) is unused: the psum-layout GEMM gets the per-
        # expert cumulative-end `psum` (built below) and unpermute uses inv_perm.
        a1q, a1q_scale, _, inv_perm, expert_num_tokens = \
            deepgemm_moe_permute(
                aq=a1q,
                aq_scale=a1q_scale,
                topk_ids=topk_ids,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                expert_tokens_meta=expert_tokens_meta,
                aq_out=a1q_perm,
                # SM100: a1 SF goes out pre-packed (UE8M0 int32 M-major) so
                # the grouped GEMM skips its transpose_and_pack glue. SM90's
                # 1d2d kernels require fp32 SF (layout.hpp asserts kFloat), so
                # packing is gated to SM100.
                scale_packed=_sf_packed,
            )
        assert a1q.size(0) == M_sum

        # Build psum layout: cumulative aligned end positions per expert.
        # psum[e] = sum_{i<=e} align128(counts[i]). Shape [E_local], GPU int32.
        # DeepGEMM's MGroupedContiguousWithPsumLayout scheduler reads psum[e]
        # and only schedules CTAs for valid groups — padding tiles are skipped.
        # Single fused triton launch (replaces add/div/mul/cumsum/cast chain).
        _align = get_mk_alignment_for_contiguous_layout()[0]
        counts, psum = expert_aligned_psum(expert_num_tokens, _align)

        mm1_out = _resize_cache(workspace2, (M_sum, N))
        m_grouped_fp8_gemm_nt_contiguous(
            (a1q, a1q_scale), (w1, self.w1_scale), mm1_out, psum,
            use_psum_layout=True, expected_m_for_psum_layout=M_sum,
        )

        # --- Fused grouped PolyNorm + 1x128 FP8 requant (count-persistent).
        # The kernel processes only valid rows per expert (skips padding).
        # Output layout (e4m3 + column-major scales) is unchanged. ---
        gate_up = mm1_out.view(-1, N)
        pn_weight, pn_bias = self._pn_params_f32()
        hc = float(self.hidden_clamp) if self.hidden_clamp is not None else -1.0
        # use_ue8m0=True always; packed_scale only on SM100 (SM90 grouped
        # kernels take fp32 SF), mirroring the a1 path.
        a2q, a2q_scale = torch.ops._C.grouped_poly_norm_fp8_quant(
            gate_up, pn_weight, pn_bias, counts,
            self.eps, hc, self.polynorm_output_scale,
            True, _sf_packed,
        )

        mm2_out = _resize_cache(workspace2, (M_sum, K))
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, psum,
            use_psum_layout=True, expected_m_for_psum_layout=M_sum,
        )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class MotifDeepGemmMoEMethod(_MotifPolyNormMoEMethodBase):
    """MoE method that converts bf16 weights to fp8 1x128 blocks at load and
    dispatches through ``MotifDeepGemmExperts`` (DeepGEMM grouped GEMM).

    Mirrors ``MotifMxfp8MoEMethod`` but targets the DeepGEMM 1d2d recipe
    (128x128 block weights, 1x128 per-token-group activations) instead of the
    CUTLASS MXFP8 (1x32) path. Activations ship bf16 through prepare and are
    quantized inside ``MotifDeepGemmExperts.apply`` (so ``_a1``/``_a2`` advertise
    no quant here), matching the CUTLASS path's self-contained structure.
    """

    @property
    def supports_eplb(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # DeepGEMM's grouped contiguous GEMM (incl. the psum layout) supports
        # both SM90 (fp32 scales, 1d2d) and SM100 (ue8m0 packed, 1d1d) since
        # the 2025.07 refactor; is_deep_gemm_e8m0_used() below picks the scale
        # format per arch.
        if not (
            current_platform.is_cuda()
            and current_platform.has_device_capability(90)
        ):
            raise RuntimeError(
                "MotifDeepGemmMoEMethod requires SM90 (Hopper) or newer."
            )
        # The whole motif DeepGEMM path ships packed-UE8M0 scale factors end to
        # end (a1 packed in ep_scatter, a2 packed by grouped_poly_norm_fp8_quant,
        # weights pre-packed below) — exponent packing is only valid for UE8M0
        # power-of-two scales, so E8M0 is required, not optional.
        assert is_deep_gemm_e8m0_used(), (
            "MotifDeepGemmMoEMethod requires DeepGEMM UE8M0 scales "
            "(do not set VLLM_USE_DEEP_GEMM_E8M0=0)."
        )

        w13 = layer.w13_weight.data  # [E_local, 2*I, K] bf16
        w2 = layer.w2_weight.data    # [E_local, K, I]   bf16
        assert w13.dtype in (torch.bfloat16, torch.float16), (
            f"expected bf16/fp16 w13 at load time, got {w13.dtype}"
        )
        w13_fp8, w13_scale = _block_quant_experts_to_fp8(w13, use_ue8m0=True)
        w2_fp8, w2_scale = _block_quant_experts_to_fp8(w2, use_ue8m0=True)

        replace_parameter(layer, "w13_weight", w13_fp8)
        replace_parameter(layer, "w2_weight", w2_fp8)
        # The *_weight_scale params stay fp32 [E, N/128, K/128]: that is the
        # layout RL refit (NeMo-RL expert_refit.refit_experts) re-quantizes
        # into via `scale.data.copy_()`. The packed-UE8M0 layout the GEMM
        # consumes is a DERIVED buffer maintained by refresh_packed_scales.
        layer.register_parameter(
            "w13_weight_scale", nn.Parameter(w13_scale, requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_scale", nn.Parameter(w2_scale, requires_grad=False),
        )
        self.refresh_packed_scales(layer)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def refresh_packed_scales(self, layer: nn.Module) -> None:
        """(Re)pack the fp32 ``*_weight_scale`` params into DeepGEMM's
        packed-UE8M0 B-side layout (skips the per-call transpose_and_pack +
        aten glue inside the grouped GEMM; ~32x the fp32 SF bytes,
        ~6MB/layer — negligible).

        Called once from process_weights_after_loading (build), and again by
        NeMo-RL's post-refit hook after expert_refit copies new fp32 scales
        into the params. The rebuild copies IN PLACE: captured cudagraphs
        hold the packed buffers' addresses and never re-enter python, so
        rebinding to fresh tensors would silently replay stale scales."""
        from vllm.utils.deep_gemm import transform_sf_into_required_layout

        for wname, sname in (
            ("w13_weight", "w13_weight_scale"),
            ("w2_weight", "w2_weight_scale"),
        ):
            src = getattr(layer, sname, None)
            wq = getattr(layer, wname, None)
            if src is None or wq is None or src.dtype != torch.float32:
                continue
            packed = transform_sf_into_required_layout(
                src.data, mn=wq.size(1), k=wq.size(2),
                recipe=(128, 128), num_groups=wq.size(0),
            )
            pname = sname + "_packed"
            existing = getattr(layer, pname, None)
            if existing is None:
                layer.register_buffer(pname, packed, persistent=False)
            else:
                existing.copy_(packed)

    def get_fused_moe_quant_config(
        self, layer: nn.Module
    ) -> FusedMoEQuantConfig | None:
        # _a1/_a2 dtype=None: bf16 tokens ship untouched; MotifDeepGemmExperts
        # quantizes the input to fp8 1x128 inside apply (before the permute
        # replicates across top_k).
        # The GEMM consumes the PACKED weight-SF buffers (derived, refreshed
        # in place on refit by refresh_packed_scales) — the fp32
        # *_weight_scale params exist as the RL-refit copy target.
        block_shape = GroupShape(_BLOCK, _BLOCK)
        a_desc = FusedMoEQuantDesc(dtype=_FP8_DTYPE, shape=block_shape)
        w1_desc = FusedMoEQuantDesc(
            dtype=_FP8_DTYPE, shape=block_shape,
            scale=layer.w13_weight_scale_packed,
        )
        w2_desc = FusedMoEQuantDesc(
            dtype=_FP8_DTYPE, shape=block_shape,
            scale=layer.w2_weight_scale_packed,
        )
        return FusedMoEQuantConfig(
            _a1=a_desc, _a2=a_desc, _w1=w1_desc, _w2=w2_desc,
            is_nvfp4_scale_swizzled=False,
        )

    def select_gemm_impl(self, prepare_finalize, layer):
        assert self.moe_quant_config is not None
        assert (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.Standard
        ), (
            "MotifDeepGemmMoEMethod only supports Standard activation format "
            f"(ag_rs), got {prepare_finalize.activation_format}"
        )
        experts = MotifDeepGemmExperts(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            poly_norm_weight=self.poly_norm_weight,
            poly_norm_bias=self.poly_norm_bias,
            hidden_clamp=self.hidden_clamp,
            polynorm_output_scale=self.polynorm_output_scale,
            polynorm_sigmoid_weight=self.polynorm_sigmoid_weight,
        )
        self._experts_impl = experts
        return experts

    def apply(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "MotifDeepGemmMoEMethod.apply should not be called; the modular "
            "kernel dispatches through MotifDeepGemmExperts."
        )
