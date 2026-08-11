# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motif-specific FusedMoE expert kernel (Standard activation format).

`MotifTritonExperts` mirrors `TritonExperts` (vLLM's default Standard-format
GEMM impl) but replaces the silu/gelu activation between GEMM1 and GEMM2 with
a per-expert grouped PolyNorm CUDA kernel. Routing, dispatch/combine,
topk-weighted reduction (via GEMM2's `mul_routed_weights` + `moe_sum`), and
CUDA graph capture are inherited from FusedMoE.

`MotifMoEMethod` is a thin `UnquantizedFusedMoEMethod` subclass that injects
`MotifTritonExperts` via `select_gemm_impl`, compatible with any Standard-
format prepare/finalize backend (deepep_high_throughput, naive,
allgather_reducescatter, flashinfer_all2allv, mori).
"""

from __future__ import annotations


import torch
import torch.nn as nn
import triton.language as tl


import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.platforms import current_platform


class _MotifPolyNormExpertsBase(mk.FusedMoEExpertsModular):
    """Shared base for Motif MoE experts (bf16 Triton and MXFP8 CUTLASS).

    Holds the per-expert PolyNorm parameters, the grouped-PolyNorm activation
    helper applied between GEMM1 and GEMM2, and the Standard-format capability
    flags common to both implementations. Subclasses provide
    `_supports_current_device`, `workspace_shapes`, and `apply`.
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
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.poly_norm_weight = poly_norm_weight
        self.poly_norm_bias = poly_norm_bias
        self.hidden_clamp = hidden_clamp
        self.polynorm_output_scale = polynorm_output_scale
        self.polynorm_sigmoid_weight = polynorm_sigmoid_weight
        self.eps = eps

    @staticmethod
    def activation_format() -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        # Motif owns its scheme; oracle gating is not used.
        return (weight_key, activation_key) == (None, None)

    @staticmethod
    def _supports_activation(_: MoEActivation) -> bool:
        # Motif owns its activation (PolyNorm); upstream enum is unused.
        return True

    @staticmethod
    def _supports_parallel_config(_: FusedMoEParallelConfig) -> bool:
        return True

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # GEMM2 / apply folds topk_weights and reduces over the topk dim —
        # finalize is a no-op.
        return TopKWeightAndReduceNoOP()

    def _grouped_polynorm_activation(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        group_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        top_k: int,
    ) -> torch.Tensor:
        """Per-expert grouped PolyNorm activation between the two GEMMs.

        The CUDA kernel folds gate/up clamp + polynom + mul into one fused
        pass; we then clamp the output (when hidden_clamp > 0) and apply the
        output scale, matching the TP path. Callers supply the per-row group
        ids, the optional EP `expert_map`, and `top_k` (kernel layout differs
        between the batched bf16 and permuted MXFP8 paths), plus
        already-contiguous gate/up.
        """
        weight = self.poly_norm_weight.float()
        if self.polynorm_sigmoid_weight:
            weight = torch.sigmoid(weight)
        bias = self.poly_norm_bias.float()
        hc = float(self.hidden_clamp) if self.hidden_clamp is not None else -1.0
        act_out = torch.ops._C.grouped_poly_norm_forward(
            gate, up, weight, bias, group_ids, expert_map, top_k, self.eps, hc,
        )
        if hc > 0:
            act_out = act_out.clamp(-hc, hc)
        if self.polynorm_output_scale != 1.0:
            act_out = act_out * self.polynorm_output_scale
        return act_out


class MotifTritonExperts(_MotifPolyNormExpertsBase):
    """Standard-format experts that use per-expert grouped PolyNorm as the
    activation between the two batched-token GEMMs.

    Layout (Standard format, mirrors `TritonExperts`):
      hidden_states         : [num_tokens, K]
      intermediate_cache1   : [num_tokens, top_k, 2*I]   (workspace2)
      intermediate_cache2   : [num_tokens * top_k, I]    (workspace13)
      intermediate_cache3   : [num_tokens, top_k, K]     (workspace2)
      output                : [num_tokens, K]
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
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            poly_norm_weight=poly_norm_weight,
            poly_norm_bias=poly_norm_bias,
            hidden_clamp=hidden_clamp,
            polynorm_output_scale=polynorm_output_scale,
            polynorm_sigmoid_weight=polynorm_sigmoid_weight,
            eps=eps,
        )
        # bf16/fp16 only: quantized MoE never goes through this class —
        # block-FP8 is MotifDeepGemmMoEMethod, MXFP8 is MotifMxfp8Experts.
        assert (
            quant_config.weight_quant_dtype is None
            and quant_config.ocp_mx_scheme is None
        ), "MotifTritonExperts: only unquantized (bf16/fp16) weights supported"

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda()

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
        # Mirror TritonExperts.workspace_shapes (PolyNorm halves the activation
        # dim, same as silu_and_mul).
        activation_out_dim = N // 2
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
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
        assert w1.stride(-1) == 1 and w2.stride(-1) == 1
        assert hidden_states.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        )

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        assert N % 2 == 0, f"intermediate size N={N} must be even (gate+up)"
        I = N // 2

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        compute_type = {
            torch.bfloat16: tl.bfloat16,
            torch.float16: tl.float16,
            torch.float32: tl.float32,
        }[hidden_states.dtype]

        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, I)
        )
        intermediate_cache3 = _resize_cache(
            workspace2, (num_tokens, top_k_num, K)
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(
                topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
            )
        )

        # GEMM1: hidden_states @ w1 → intermediate_cache1 [num_tokens, top_k, 2*I]
        invoke_fused_moe_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            self.w1_scale,
            None,  # topk_weights — applied at GEMM2
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w1_bias,
        )

        # ----- Activation: per-expert grouped PolyNorm -----
        # Reshape ic1 to [num_tokens*top_k, 2*I], split gate / up.
        ic1_flat = intermediate_cache1.view(-1, N)
        gate = ic1_flat[:, :I].contiguous()
        up = ic1_flat[:, I:].contiguous()

        # Kernel expects int32 (Tensor.to is a no-op when dtype already matches).
        topk_ids_i32 = topk_ids.to(torch.int32)
        emap_i32 = expert_map.to(torch.int32) if expert_map is not None else None
        act_out = self._grouped_polynorm_activation(
            gate, up, topk_ids_i32, emap_i32, top_k_num
        )

        intermediate_cache2.copy_(act_out.to(intermediate_cache2.dtype))

        # GEMM2: intermediate_cache2 @ w2 → intermediate_cache3 [num_tokens, top_k, K]
        # `mul_routed_weights=True` folds topk_weights into GEMM2.
        invoke_fused_moe_triton_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            a2_scale,
            self.w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w2_bias,
        )

        # Reduce over topk dim → [num_tokens, K] with fp32 accumulator to
        # match TP path (motif.py:862-863 scatter_add into fp32 buffer).
        # csrc moe_sum_kernel uses scalar_t (bf16) accumulator and loses
        # precision when summing top_k contributions across 32 MoE layers.
        output.copy_(
            intermediate_cache3.sum(dim=1, dtype=torch.float32).to(output.dtype)
        )


class MoEPrepareAndFinalizeIdentityAllReduce(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/finalize for TP-only Motif MoE serving (no DP, no all2all).

    Each TP rank already holds the full `[T, H]` MoE input (TP attention
    output is replicated across the TP world via attention's all_reduce).
    Each rank stores `num_experts // tp_size` experts and computes only for
    tokens routed to its local experts (filtered by FusedMoE's `expert_map`).
    Cross-rank summation happens in `finalize` via a single
    `tensor_model_parallel_all_reduce` of the `[T, H]` partial output buffer
    — mirrors the prior `MotifMoEExperts` semantics (motif.py:862-870).

    No dispatch all2all, no combine all2all. Compile-friendly: the only
    collective is `torch.ops.vllm.all_reduce` (registered op, fake_impl-aware).
    """

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return 1

    def output_is_reduced(self) -> bool:
        # `finalize()` ends in tensor_model_parallel_all_reduce, so the output
        # is fully reduced across the TP world group.
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        # Identity dispatch: input is already replicated at every TP rank.
        # MotifTritonExperts is bf16-only (asserted), so no quantization.
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)
        return a1, None, None, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        # Reduce over the topk dim into a fresh fp32 buffer (mirrors TP path's
        # fp32 promotion at motif.py:862-868).
        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        if out.dtype != torch.float32:
            out = out.float()
        # Single cross-rank all_reduce: each rank contributes only its
        # local-expert outputs (others are zero), so the sum is correct.
        out = tensor_model_parallel_all_reduce(out)
        output.copy_(out.to(output.dtype))


class _MotifPolyNormMoEMethodBase(UnquantizedFusedMoEMethod):
    """Shared base for Motif MoE methods (bf16 `MotifMoEMethod` and MXFP8
    `MotifMxfp8MoEMethod`).

    Holds the per-expert PolyNorm parameters and the prepare/finalize wiring
    (TP-only identity all-reduce vs naive DP/EP with optional fp32 combine)
    that is identical across both methods. Subclasses provide
    `process_weights_after_loading`, `select_gemm_impl`, and (MXFP8 only)
    `get_fused_moe_quant_config`.
    """

    def __init__(
        self,
        moe: FusedMoEConfig,
        poly_norm_weight: torch.Tensor,
        poly_norm_bias: torch.Tensor,
        hidden_clamp: float | None,
        polynorm_output_scale: float,
        polynorm_sigmoid_weight: bool,
        use_identity_allreduce: bool = False,
    ):
        super().__init__(moe)
        self.poly_norm_weight = poly_norm_weight
        self.poly_norm_bias = poly_norm_bias
        self.hidden_clamp = hidden_clamp
        self.polynorm_output_scale = polynorm_output_scale
        self.polynorm_sigmoid_weight = polynorm_sigmoid_weight
        self.use_identity_allreduce = use_identity_allreduce
        # Force the modular-kernel construction path; disable the monolithic
        # shortcut so select_gemm_impl is reached for every backend.
        self._is_monolithic = False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # TP-only Motif path: skip all2all backends entirely. Input is already
        # replicated across TP ranks (TP attn output), and combine is a single
        # tensor_model_parallel_all_reduce of the partial output buffer.
        if self.use_identity_allreduce:
            return MoEPrepareAndFinalizeIdentityAllReduce()

        # vLLM's default UnquantizedFusedMoEMethod path returns None for the
        # naive / allgather_reducescatter all2all backends because the
        # unquantized MoE migration to the new modular-kernel interface is
        # incomplete (all2all_utils.py:96-114 / 205-210 gates it on
        # `allow_new_interface=True`, only set by the FP8/NVFP4/MXFP4 oracles).
        # When None is returned, no FusedMoEModularMethod is created, so
        # select_gemm_impl is never called and MotifTritonExperts is never
        # instantiated — routed experts silently fall back to the default
        # TritonExperts (silu_and_mul), bypassing PolyNorm entirely.
        # Force the modular path here so MotifTritonExperts is wired in for
        # every Standard-format backend (naive / allgather_reducescatter /
        # deepep_high_throughput / flashinfer_all2allv).
        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize as _make,
        )
        from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (  # noqa: E501
            MoEPrepareAndFinalizeNaiveDPEPModular,
        )

        pf = _make(
            self.moe,
            self.moe_quant_config,
            routing_tables,
            allow_new_interface=True,
        )

        # bf16 reduce-scatter written directly into the output buffer: stock
        # finalize materializes the RS result and then copies it into output,
        # and the partials entering the combine are already bf16, so there is
        # nothing for an fp32 round-trip to protect (see the class docstring).
        if isinstance(pf, MoEPrepareAndFinalizeNaiveDPEPModular):
            pf = _get_direct_combine_pf_cls()(
                is_sequence_parallel=pf.is_sequence_parallel,
                num_dispatchers=pf.num_dispatchers(),
            )
        return pf


class MotifMoEMethod(_MotifPolyNormMoEMethodBase):
    """FusedMoE method that injects `MotifTritonExperts` via select_gemm_impl.

    Per-expert PolyNorm parameters (`poly_norm_weight [E_local, 3]`,
    `poly_norm_bias [E_local, 1]`) are owned by the caller and passed into the
    experts class at construction.

    `use_identity_allreduce=True` switches the prepare/finalize backend to
    `MoEPrepareAndFinalizeIdentityAllReduce` for the TP-only Motif path
    (no DP, no all2all dispatch — single TP world all_reduce).
    """

    def process_weights_after_loading(self, layer):
        # Skip UnquantizedFusedMoEMethod._setup_kernel: we don't use the
        # default unquantized MoE kernel — the modular kernel built around
        # MotifTritonExperts (via select_gemm_impl) is the only execution
        # path. _setup_kernel would also call convert_to_unquantized_kernel_format,
        # which under VLLM_USE_FLASHINFER_MOE_FP16=1 swaps w13->w31 and would
        # silently break MotifTritonExperts' gate/up split. Only initialize
        # the moe_quant_config that select_gemm_impl asserts on.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def select_gemm_impl(self, prepare_finalize, layer):
        assert (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.Standard
        ), (
            "MotifMoEMethod requires Standard-format prepare/finalize. "
            "Use --all2all-backend deepep_high_throughput / naive / "
            "allgather_reducescatter / flashinfer_all2allv."
        )
        assert self.moe_quant_config is not None
        return MotifTritonExperts(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            poly_norm_weight=self.poly_norm_weight,
            poly_norm_bias=self.poly_norm_bias,
            hidden_clamp=self.hidden_clamp,
            polynorm_output_scale=self.polynorm_output_scale,
            polynorm_sigmoid_weight=self.polynorm_sigmoid_weight,
        )


# Lazy-built to avoid circular imports: prepare_finalize.naive_dp_ep transitively
# pulls in modules that may not be ready when motif_experts.py is first imported.
_direct_combine_pf_cls: type | None = None


def _get_direct_combine_pf_cls():
    global _direct_combine_pf_cls
    if _direct_combine_pf_cls is not None:
        return _direct_combine_pf_cls

    from vllm.distributed import get_dp_group, get_ep_group
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (  # noqa: E501
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )

    class MotifDirectCombineNaivePrepareFinalize(
        MoEPrepareAndFinalizeNaiveDPEPModular
    ):
        """Naive DP/EP prepare/finalize: bf16 reduce-scatter straight into
        the caller's output buffer.

        Replaces the earlier fp32-promoted combine. The per-rank partials are
        ALREADY bf16 (the experts' fused_out workspace is act-dtype; ep_gather
        accumulates in fp32 registers and stores bf16), so the fp32 round-trip
        only protected the 8-way cross-rank ADD arithmetic on bf16-rounded
        operands — at 2x the reduce-scatter bytes plus a full-size fp32->bf16
        cast copy per MoE layer. Reduce-scatter in bf16 (standard TP
        all-reduce practice) and write the result directly into ``output``,
        eliminating both.
        """

        def finalize(
            self,
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        ):
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            out = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            # Mirrors NaiveAll2AllManager.combine's reduce_scatterv, with the
            # collective writing into ``output`` (out=) instead of returning a
            # fresh tensor that finalize then copies.
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
            assert sizes is not None
            dist_group = (
                get_ep_group() if self.is_sequence_parallel else get_dp_group()
            )
            assert output.shape[0] == sizes[dist_group.rank_in_group]
            assert output.dtype == out.dtype, (
                f"direct combine dtype mismatch: {output.dtype} vs {out.dtype}"
            )
            dist_group.reduce_scatterv(out, dim=0, sizes=sizes, out=output)

    _direct_combine_pf_cls = MotifDirectCombineNaivePrepareFinalize
    return _direct_combine_pf_cls
