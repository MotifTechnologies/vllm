# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Motif model with GDLA (Grouped Differential Latent Attention) + MHC."""

import math
import os
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.v1.attention.backends.flash_attn_diffkv import FlashAttentionDiffKVBackend
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE as SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.motif import MotifConfig
from vllm.utils.torch_utils import direct_register_custom_op

from .interfaces import SupportsPP
from .utils import (
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from loguru import logger


# ---------------------------------------------------------------------------
# Vendored Triton kernels for MHC (forward-only port of TT's training kernels:
# sinkhorn_fused, res_triton, mhc_post_fused). See motif_mhc_kernels.py.
# ---------------------------------------------------------------------------
try:
    from vllm.model_executor.layers.motif_mhc_kernels import (
        sinkhorn_fused as _tt_sinkhorn_fused,
        res_triton as _tt_res_triton,
        mhc_post_fused as _tt_mhc_post_fused,
    )
    _HAS_TT_MHC_KERNELS = True
except ImportError:
    _HAS_TT_MHC_KERNELS = False

def _stat(tag: str, x: torch.Tensor) -> None:
    """Activation statistics logger (no-op; enable by uncommenting body)."""
    return None


# ---------------------------------------------------------------------------
# PolyNorm activation (trainable polynomial normalization)
# ---------------------------------------------------------------------------

# Inlined fp32 PolyNorm body — separated so it can be compiled as one fullgraph
# region (no TP all-reduce, no sigmoid_weight branch). Inductor fuses pow/mean
# per branch into reduce kernels and rsqrt/mul/add into pointwise kernels,
# replacing ~12 unfused at::native::* launches per call with ~3 triton kernels.
@torch.compile(fullgraph=True, dynamic=True)
def _poly_norm_compute(
    x: torch.Tensor,        # any shape [..., H], any dtype (cast inside)
    w: torch.Tensor,        # [3] fp32 (post-sigmoid if sigmoid_weight)
    b: torch.Tensor,        # [1] fp32
    eps: float,
    mul: torch.Tensor | None = None,   # optional [..., H] to multiply (fp32)
) -> torch.Tensor:
    orig_dtype = x.dtype
    g = x.float()
    g2 = g * g
    g3 = g2 * g
    inv2 = torch.rsqrt(g2.mean(-1, keepdim=True) + eps)
    inv4 = torch.rsqrt((g2 * g2).mean(-1, keepdim=True) + eps)
    inv6 = torch.rsqrt((g3 * g3).mean(-1, keepdim=True) + eps)
    poly = (w[0] * g3 * inv6
          + w[1] * g2 * inv4
          + w[2] * g  * inv2
          + b)
    if mul is not None:
        # Multiply by `up` in fp32 and downcast once at the end, matching
        # training's FusedMulPolyNorm (grouped_polynorm.py: `poly * m` in fp32,
        # `.to(orig_dtype)` last). Downcasting poly first would lose ~1e-3.
        return (poly * mul.float()).to(orig_dtype)
    return poly.to(orig_dtype)


# MLP-fused PolyNorm: clamp(gate) + clamp(up) + PolyNorm(gate) * up + scale,
# all in one compile region so inductor fuses the clamps and the *up multiply
# with the polynorm pointwise stage. Compiled with the same @torch.compile
# settings; hidden_clamp / output_scale are static per-layer constants so
# Dynamo specializes on them.
@torch.compile(fullgraph=True, dynamic=True)
def _poly_norm_mlp_compute(
    gate: torch.Tensor,      # [..., H] gate_proj output
    up: torch.Tensor,        # [..., H] up_proj output
    w: torch.Tensor,         # [3] fp32 (cached, post-sigmoid)
    b: torch.Tensor,         # [1] fp32
    eps: float,
    hidden_clamp: float,
    output_scale: float,
) -> torch.Tensor:
    gate = gate.clamp(-hidden_clamp, hidden_clamp)
    up = up.clamp(-hidden_clamp, hidden_clamp)
    g = gate.float()
    g2 = g * g
    g3 = g2 * g
    inv2 = torch.rsqrt(g2.mean(-1, keepdim=True) + eps)
    inv4 = torch.rsqrt((g2 * g2).mean(-1, keepdim=True) + eps)
    inv6 = torch.rsqrt((g3 * g3).mean(-1, keepdim=True) + eps)
    poly = (w[0] * g3 * inv6
          + w[1] * g2 * inv4
          + w[2] * g  * inv2
          + b)
    # Multiply by `up` in fp32, downcast once (matches training's
    # FusedMulPolyNorm). output_scale is applied after the downcast, as in the
    # training FeedForward (`h = fused(...); h = h * output_scale`).
    out = (poly * up.float()).to(gate.dtype)
    return out * output_scale


class PolyNormTorch(nn.Module):
    """Trainable poly-norm activation. TP=1 hot path is dispatched to a
    `@torch.compile(fullgraph=True)` helper so inductor fuses the pow/mean +
    rsqrt/mul/add chain into a small number of triton kernels. TP>1 keeps
    the eager all-reduce path (collective + compile interaction is messy and
    the TP>1 case is rare for this model).
    """

    def __init__(self, eps: float = 1e-6, sigmoid_weight: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3, dtype=torch.float32) / 3)
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.eps = eps
        self.sigmoid_weight = sigmoid_weight
        self._tp_size = get_tensor_model_parallel_world_size()
        # Lazy sigmoid(weight) cache for the eager path (shared experts run
        # inside the opaque FusedMoE op): saves a cast+sigmoid launch per
        # call. Weights are static during serving; an RL refit (in-place
        # weight push) must call refresh_weight_cache() afterwards — NeMo-RL
        # does this from its post-refit hook (see motif_quickopt refit
        # module), mirroring the tilelang MHC graft. Compiled callers
        # (dense-MLP layers) bypass the cache — inductor fuses the sigmoid.
        self._w_cache: torch.Tensor | None = None

    def _compute_weight(self) -> torch.Tensor:
        w = self.weight.float()
        return torch.sigmoid(w) if self.sigmoid_weight else w

    def _get_weight(self) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self._compute_weight()
        if self._w_cache is None:
            self._w_cache = self._compute_weight()
        return self._w_cache

    def refresh_weight_cache(self) -> None:
        """RL-refit hook: recompute the cached fold IN PLACE (cudagraphs
        capture the cache tensor's address). No-op if never built — the lazy
        path then builds fresh from the refit weights."""
        if self._w_cache is not None:
            self._w_cache.copy_(self._compute_weight())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._tp_size > 1:
            return self._forward_tp(x)
        return _poly_norm_compute(x, self._get_weight(), self.bias, self.eps)

    def forward_mul(self, x: torch.Tensor, mul: torch.Tensor) -> torch.Tensor:
        """PolyNorm(x) * mul with the multiply done in fp32 before the single
        downcast, matching training's FusedMulPolyNorm. Used by MotifMLP so the
        `gate * up` product keeps fp32 precision (downcasting PolyNorm(x) first,
        as a bare `forward(x) * up`, would lose ~1e-3)."""
        if self._tp_size > 1:
            return self._forward_tp(x, mul)
        return _poly_norm_compute(x, self._get_weight(), self.bias, self.eps, mul)

    def _forward_tp(
        self, x: torch.Tensor, mul: torch.Tensor | None = None
    ) -> torch.Tensor:
        g = x.float()
        g2, g3 = g * g, g * g * g
        local_vars = torch.stack([
            g.pow(2).mean(-1), g.pow(4).mean(-1), g.pow(6).mean(-1)
        ], dim=-1)
        gv = tensor_model_parallel_all_reduce(local_vars) / self._tp_size
        w = self._get_weight()
        b = self.bias
        poly = (w[0] * g3 / torch.sqrt(gv[:, 2:3] + self.eps)
              + w[1] * g2 / torch.sqrt(gv[:, 1:2] + self.eps)
              + w[2] * g  / torch.sqrt(gv[:, 0:1] + self.eps)
              + b)
        if mul is not None:
            return (poly * mul.float()).to(x.dtype)
        return poly.to(x.dtype)


# ---------------------------------------------------------------------------
# MHC Layer (Manifold-constrained Hyper-Connections)
# ---------------------------------------------------------------------------

class MotifMHCLayer(nn.Module):
    """MHC layer from https://arxiv.org/abs/2512.24880.

    Operates on tensors of shape (1, num_tokens, E, D) where E = expansion_rate.
    """

    def __init__(
        self,
        expansion_rate: int,
        num_dim: int,
        identity_init: bool = False,
        sinkhorn_iters: int = 20,
        h_post_coeff: float = 2.0,
        use_tilelang: bool = False,
    ):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.num_dim = num_dim
        self.sinkhorn_iters = sinkhorn_iters
        # H_post = h_post_coeff * sigmoid(raw). Training uses (1 + alpha)*sigmoid;
        # at inference, coeff = 1 + mhc_h_post_alpha_end from the trained schedule.
        self.h_post_coeff = float(h_post_coeff)

        E, D = expansion_rate, num_dim
        # Single merged projection [pre(E) | post(E) | res(E²)] matching torchtitan's
        # `proj_merged` layout. We're NOT actually doing TP here — MHC has no TP
        # sharding — but vLLM has no plain `MergedLinear`, and we want the built-in
        # per-shard weight_loader so load_weights() can route the per-shard
        # checkpoint tensors (proj_pre/post/res.weight) into the right rows via
        # shard ids. `disable_tp=True` is just the way to opt out of the TP infra
        # this class otherwise provides.
        self.proj_merged = MergedColumnParallelLinear(
            input_size=E * D,
            output_sizes=[E, E, E * E],
            bias=False,
            disable_tp=True,
            return_bias=False,
        )
        self.rms_norm = RMSNorm(E * D, eps=1e-6)

        self.bias_pre = nn.Parameter(torch.empty(E, dtype=torch.float32))
        self.bias_post = nn.Parameter(torch.empty(E, dtype=torch.float32))
        self.bias_res = nn.Parameter(torch.empty(E, E, dtype=torch.float32))
        self.alpha_pre = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.alpha_post = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.alpha_res = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self._init_weights(identity_init)

        # Opt-in tilelang/DeepGEMM MHC path (DeepSeek-V4 kernels). Importing the
        # module registers torch.ops.vllm.mhc_pre / mhc_post and hard-requires
        # `tilelang` (raises ImportError with an install hint if absent).
        self._use_tilelang = bool(use_tilelang)
        self._fused_built = False
        if self._use_tilelang:
            import vllm.model_executor.layers.mhc  # noqa: F401

    def _init_weights(self, identity_init: bool) -> None:
        if identity_init:
            nn.init.zeros_(self.alpha_pre)
            nn.init.zeros_(self.alpha_post)
            nn.init.zeros_(self.alpha_res)
            nn.init.xavier_uniform_(self.proj_merged.weight)
            uniform_weight = 1.0 / self.expansion_rate
            v = math.log(uniform_weight / (1 - uniform_weight)) if 0 < uniform_weight < 1 else 0.0
            nn.init.constant_(self.bias_pre, v)
            nn.init.zeros_(self.bias_post)
            nn.init.constant_(self.bias_res, -10.0)
            self.bias_res.data.fill_diagonal_(0.0)
        else:
            nn.init.normal_(self.alpha_pre, std=0.1)
            nn.init.normal_(self.alpha_post, std=0.1)
            nn.init.normal_(self.alpha_res, std=0.1)
            nn.init.xavier_uniform_(self.proj_merged.weight)
            nn.init.zeros_(self.bias_pre)
            nn.init.zeros_(self.bias_post)
            nn.init.normal_(self.bias_res, std=0.1)

    @torch.no_grad()
    def build_fused_params(self) -> None:
        """Precompute fn / hc_scale / hc_base buffers for torch.ops.vllm.mhc_pre
        from the trained proj_merged / rms_norm / alpha_* / bias_* params.

        Idempotent; called lazily before the first forward (outside the
        compiled region). RMS gamma is folded into fn columns (mhc_pre's
        internal RMS is weight-free); fn rows keep proj_merged's
        [pre(E) | post(E) | res(E*E)] layout.

        NOTE (RL refit contract): NeMo-RL's tilelang_mhc.refit_fused_params
        refreshes these buffers after a weight refit by resetting
        ``_fused_built``, re-calling this, and grafting the fresh values into
        the ORIGINAL buffer storage (cudagraphs capture those addresses). It
        keys off ``_fused_built`` / ``_mhc_fused_ready`` / ``_use_tilelang``
        and the ``_fn``/``_hc_scale``/``_hc_base`` names — do not rename.
        """
        if self._fused_built:
            return
        W = self.proj_merged.weight.data.to(torch.float32)      # (E*E+2E, E*D)
        gamma = self.rms_norm.weight.data.to(torch.float32)     # (E*D,)
        fn = (W * gamma.unsqueeze(0)).contiguous()              # fold RMS gamma
        hc_scale = torch.stack([
            self.alpha_pre.data.reshape(()),
            self.alpha_post.data.reshape(()),
            self.alpha_res.data.reshape(()),
        ]).to(torch.float32).contiguous()
        hc_base = torch.cat([
            self.bias_pre.data.reshape(-1),
            self.bias_post.data.reshape(-1),
            self.bias_res.data.reshape(-1),                    # direct layout
        ]).to(torch.float32).contiguous()
        self.register_buffer("_fn", fn, persistent=False)
        self.register_buffer("_hc_scale", hc_scale, persistent=False)
        self.register_buffer("_hc_base", hc_base, persistent=False)
        self._rms_eps = float(getattr(self.rms_norm, "variance_epsilon", 1e-6))
        self._fused_built = True

    def mhc_pre_tilelang(self, x_3d: torch.Tensor):
        """x_3d: (T, E, D) bf16 -> (post_mix (T,E,1), comb (T,E,E), layer_input
        (T,D)). Fused proj + RMS + sigmoid + sinkhorn + apply_h_pre.

        With motif_sinkhorn=1 the kernel uses the training sinkhorn convention
        and stores comb as sinkhorn(M)^T, so mhc_post's comb^T application
        yields exactly sinkhorn(M) @ x like the Triton path.
        """
        return torch.ops.vllm.mhc_pre(
            x_3d,
            self._fn,
            self._hc_scale,
            self._hc_base,
            self._rms_eps,        # rms_eps
            0.0,                  # hc_pre_eps: motif adds no eps to h_pre
            1e-8,                 # hc_sinkhorn_eps == Triton clamp-min
            self.h_post_coeff,    # hc_post_mult_value
            self.sinkhorn_iters,  # sinkhorn_repeat == motif sinkhorn_iters
            1,                    # n_splits (mhc_pre recomputes internally)
            1,                    # motif_sinkhorn (training convention)
        )

    def _sinkhorn(self, m: torch.Tensor) -> torch.Tensor:
        if _HAS_TT_MHC_KERNELS and m.is_cuda:
            return _tt_sinkhorn_fused(m.float().contiguous(), self.sinkhorn_iters)
        orig_dtype = m.dtype
        m = m.float().clamp(-20.0, 20.0).exp()
        for _ in range(self.sinkhorn_iters):
            m = m / m.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            m = m / m.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        return m.to(orig_dtype)

    def forward(self, x: torch.Tensor):
        # x: (1, T, E, D)
        _, T, E, D = x.shape
        if not torch.compiler.is_compiling() and T == 0:
            # Guard for dummy profiling runs with empty batches.
            # Compile-time dead-coded: Dynamo specializes is_compiling()=True
            # so this branch is removed from the traced graph.
            return (
                x.new_zeros(1, 0, E),
                x.new_zeros(1, 0, E),
                x.new_zeros(1, 0, E, E),
            )

        x_flat = x.reshape(1, T, E * D).contiguous()
        x_norm = self.rms_norm(x_flat.reshape(T, E * D)).reshape(T, E * D).contiguous()

        # Single merged GEMM (matches torchtitan layout: [pre(E) | post(E) | res(E²)] rows).
        proj_all = self.proj_merged(x_norm).float()  # [T, E²+2E]
        pre  = proj_all[:, :E].reshape(1, T, E)
        post = proj_all[:, E : 2 * E].reshape(1, T, E)
        res  = proj_all[:, 2 * E :].reshape(1, T, E, E)

        # alpha_*/bias_* are stored as fp32 (see __init__); no .float() casts needed.
        h_pre  = torch.sigmoid((self.alpha_pre  * pre  + self.bias_pre).clamp(-10, 10))
        h_post = self.h_post_coeff * torch.sigmoid((self.alpha_post * post + self.bias_post).clamp(-10, 10))
        h_res  = self._sinkhorn(self.alpha_res * res + self.bias_res)
        return h_pre, h_post, h_res

    @staticmethod
    def apply_h_pre(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        """(1, T, E, D), (1, T, E) -> (1, T, D)."""
        return (x * h_pre.unsqueeze(-1)).sum(dim=2).to(x.dtype)

    @staticmethod
    def apply_h_post(x: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
        """(1, T, D), (1, T, E) -> (1, T, E, D)."""
        return (h_post.unsqueeze(-1) * x.unsqueeze(2)).to(x.dtype)

    @staticmethod
    def apply_h_res(x: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
        """(1, T, E, E), (1, T, E, D) -> (1, T, E, D)."""
        if _HAS_TT_MHC_KERNELS and x.is_cuda:
            return _tt_res_triton(h_res.contiguous(), x.contiguous())
        return torch.einsum("bsij,bsjd->bsid", h_res, x.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Attention (GDLA)
# ---------------------------------------------------------------------------

class MotifGDLAttention(nn.Module):
    """Grouped Differential Latent Attention for vLLM.

    Maps checkpoint weights:
      wq_a         -> q_a_proj      (ReplicatedLinear)
      q_norm       -> q_a_layernorm (RMSNorm)
      wq_b         -> q_b_proj      (ColumnParallelLinear)
      wq_b_gate    -> q_b_gate      (ColumnParallelLinear, optional)
      wkv_a        -> kv_a_proj_with_mqa (ReplicatedLinear)
      kv_norm      -> kv_a_layernorm (RMSNorm)
      wkv_b        -> kv_b_proj     (ColumnParallelLinear)
      lambda_proj  -> lambda_proj   (ColumnParallelLinear)
      wo           -> o_proj        (RowParallelLinear)
    """

    def __init__(
        self,
        config: MotifConfig,
        layer_idx: int,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim if config.head_dim is not None else self.hidden_size // self.num_heads

        self.num_noise_heads = config.num_noise_heads
        self.grouped_ratio = (self.num_heads - self.num_noise_heads) // self.num_noise_heads
        self.n_signal_heads = self.grouped_ratio * self.num_noise_heads

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim if config.qk_rope_head_dim is not None else self.head_dim // 2
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim if config.v_head_dim is not None else self.head_dim

        self.elementwise_gate = config.elementwise_attn_output_gate

        # Debug env-var snapshots (read once at init, no env-var reads in forward
        # to avoid graph breaks under torch.compile).
        self._save_gdl = bool(os.environ.get("MOTIF_SAVE_GDL")) and layer_idx == 0
        self._use_sdpa = bool(os.environ.get("MOTIF_USE_SDPA"))
        self._sdpa_math = bool(os.environ.get("MOTIF_SDPA_MATH"))
        self._trace_all = bool(os.environ.get("MOTIF_TRACE_ALL"))

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        assert self.num_kv_heads % tp_size == 0
        assert self.num_noise_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size
        self.num_local_kv_heads = self.num_kv_heads // tp_size
        self.local_noise_heads = self.num_noise_heads // tp_size
        self.local_signal_heads = self.grouped_ratio * self.local_noise_heads

        # Determine per-layer attention mode early — needed for (a) softmax
        # scale (YaRN mscale applies only to full-attention layers; SWA uses
        # plain RoPE so mscale² would be wrong there) and (b) kv_b_proj sizing +
        # attention backend selection below (MLA layers size kv_b_proj to
        # num_heads and route to MLAAttention; SWA keeps GQA sizing + the
        # diff-KV flash backend). When SWA is disabled every layer stays on the
        # standard GQA path, so non-hybrid configs behave exactly as before.
        per_layer_sliding_window = None
        is_full_attn_in_interleaved = False
        if config.use_sliding_window and config.sliding_window is not None:
            pattern = config.sliding_window_pattern
            period = config.sliding_window_period
            # Training passes flash-attn window_size=(sliding_window, 0), so a
            # query attends to itself plus `sliding_window` past keys
            # (sliding_window + 1 tokens total). vLLM's FlashAttention backend
            # converts the per-layer window to (window - 1, 0), so we add 1 here
            # to reproduce the trained window exactly (matches the HF reference's
            # `config.sliding_window + 1`). Without the +1, SWA layers attend to
            # one fewer key than training.
            swa_window = config.sliding_window + 1
            if pattern == "all":
                per_layer_sliding_window = swa_window
            elif pattern == "interleave":
                if layer_idx % period != 0:
                    per_layer_sliding_window = swa_window
                    logger.debug(f"sliding window layers : {layer_idx}")
                else:
                    is_full_attn_in_interleaved = True
        _is_swa_layer = per_layer_sliding_window is not None
        self.is_mla_layer = is_full_attn_in_interleaved

        # Softmax scale (with optional mscale).
        # YaRN mscale is applied ONLY to full-attention layers; SWA layers use
        # plain RoPE (theta=swa_rope_theta, no context extension) so mscale²
        # would be wrong for them.  Matches training code:
        #   if (not is_swa) and max_seq_len > original_seq_len: apply mscale²
        self.scaling = self.head_dim ** -0.5
        original_seq_len = getattr(config, "original_seq_len", 32768)
        rope_factor = getattr(config, "rope_factor", 1.0)
        mscale = getattr(config, "mscale", 1.0)
        if (not _is_swa_layer) and config.max_position_embeddings > original_seq_len:
            mscale_val = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.scaling = self.scaling * mscale_val * mscale_val

        # Q projections
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        if self.elementwise_gate:
            self.q_b_gate = ColumnParallelLinear(
                self.q_lora_rank,
                self.n_signal_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_gate",
            )
        else:
            self.q_b_gate = None

        # KV projections
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # MLA layers size kv_b_proj to num_heads so MLAAttention can absorb
        # W_UK/W_UV per query head; SWA layers keep the GQA num_kv_heads sizing.
        # The checkpoint always stores the GQA shape — load_weights replicates
        # it to num_heads for MLA layers (mathematically equivalent to GQA).
        kv_b_proj_heads = self.num_heads if self.is_mla_layer else self.num_kv_heads
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            kv_b_proj_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # Lambda projection in float32: sigmoid is sensitive near 0 and the
        # values are small scalars — bf16 precision loss noticeably affects
        # the signal/noise ratio in differential attention.
        self.lambda_proj = ColumnParallelLinear(
            self.hidden_size,
            self.n_signal_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lambda_proj",
        )

        # Output projection: n_signal_heads * v_head_dim -> hidden_size
        self.o_proj = RowParallelLinear(
            self.n_signal_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE (only over qk_rope_head_dim dims)
        # Motif uses neox-style rotation (rotate_half splits into halves)
        # Full-attention layers: YaRN scaling with rope_theta (e.g. 5e6 for 32k model)
        # SWA layers: plain RoPE with swa_rope_theta (e.g. 10000) — no long-ctx scaling needed
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        rope_params = {"rope_theta": config.rope_theta, **rope_scaling}
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rope_params,
        )
        self.is_swa_layer = _is_swa_layer
        swa_rope_theta = getattr(config, "swa_rope_theta", None)
        if self.is_swa_layer and swa_rope_theta is not None:
            self.swa_rotary_emb = get_rope(
                self.qk_rope_head_dim,
                max_position=config.max_position_embeddings,
                is_neox_style=True,
                rope_parameters={"rope_theta": swa_rope_theta},
            )
        else:
            self.swa_rotary_emb = None

        # Per-layer attention backend. Full-attention layers (is_mla_layer)
        # route to MLAAttention, which selects an MLA backend internally
        # (use_mla=True) and reports an MLAAttentionSpec to the engine — its KV
        # cache stores only the latent (kv_lora_rank + qk_rope_head_dim) per
        # token. SWA / non-hybrid layers keep the standard GQA path through the
        # diff-KV flash backend.
        if self.is_mla_layer:
            self.mla_attn = MLAAttention(
                num_heads=self.num_local_heads,
                scale=self.scaling,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                kv_b_proj=self.kv_b_proj,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )
            self.attn = None
        else:
            self.mla_attn = None
            # Diff K/V head_dim (e.g., qk=192, v=128 like deepseek-MLA) requires
            # a backend whose KV cache layout uses head_size + head_size_v. The
            # default FA backend assumes K=V dim, which mis-reshapes the cache.
            # mimo_v2_flash.py uses the same pattern.
            if self.v_head_dim != self.head_dim:
                FlashAttentionDiffKVBackend.set_head_size_v(self.v_head_dim)
                attn_backend = FlashAttentionDiffKVBackend
            else:
                attn_backend = None

            self.attn = Attention(
                self.num_local_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_local_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                per_layer_sliding_window=per_layer_sliding_window,
                head_size_v=self.v_head_dim,
                attn_backend=attn_backend,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        L = self.layer_idx
        _save_gdl = self._save_gdl
        _gdl = {}

        # Q path
        q_latent, _ = self.q_a_proj(hidden_states)      # [T, q_lora_rank]
        q_latent = self.q_a_layernorm(q_latent)
        if _save_gdl:
            _gdl['q_latent'] = q_latent.detach().float().cpu()
        q, _ = self.q_b_proj(q_latent)                  # [T, num_local_heads * head_dim]
        if _save_gdl:
            _gdl['q_after_wqb'] = q.detach().float().cpu()
        q = q.view(num_tokens, self.num_local_heads, self.head_dim)

        gate_score = None
        if self.elementwise_gate and self.q_b_gate is not None:
            gate_score, _ = self.q_b_gate(q_latent)     # [T, local_signal_heads * v_head_dim]
            gate_score = gate_score.view(num_tokens, self.local_signal_heads, self.v_head_dim)

        # Split Q into nope + rope dims
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV down-projection (latent + decoupled rope key) — common to both modes
        kv_raw, _ = self.kv_a_proj_with_mqa(hidden_states)  # [T, kv_lora_rank + qk_rope_head_dim]
        kv_latent, k_pe = kv_raw.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)

        # Apply RoPE to rope dims (float32 — bf16 precision loss measurably
        # affects logits); k_pe is shared across all KV heads.
        k_pe_head = k_pe.unsqueeze(1)  # [T, 1, qk_rope_head_dim]

        if _save_gdl:
            _gdl['q_pe_before_rope'] = q_pe.detach().float().cpu()
            _gdl['k_pe_before_rope'] = k_pe.detach().float().cpu()

        _rope_dtype = q_pe.dtype
        _rotary = self.swa_rotary_emb if self.swa_rotary_emb is not None else self.rotary_emb
        q_pe, k_pe_head = _rotary(positions, q_pe.float(), k_pe_head.float())
        q_pe = q_pe.to(_rope_dtype)
        k_pe_head = k_pe_head.to(_rope_dtype)

        if _save_gdl:
            _gdl['q_pe_after_rope'] = q_pe.detach().float().cpu()
            _gdl['k_pe_after_rope'] = k_pe_head.detach().float().cpu()

        # Reconstruct full Q (both modes consume this)
        q_full = torch.cat([q_nope, q_pe], dim=-1)       # [T, local_heads, qk_head_dim]

        # Lambda (one per local signal head)
        lambda_vals, _ = self.lambda_proj(hidden_states)  # [T, local_signal_heads]
        if _save_gdl:
            _gdl['lambda'] = lambda_vals.detach().float().cpu()

        # Attention: branch on per-layer mode.
        if self.is_mla_layer:
            # MLA: kv_b_proj is absorbed inside MLAAttention; the KV cache stores
            # only latent (kv_lora_rank + qk_rope_head_dim) per token. Output is
            # post-W_UV per-head v (pre-o_proj), shape (T, num_heads*V), reshaped
            # to (T, heads, V) for the signal/noise split below. k_full / v are
            # not materialized in this path.
            k_full = None
            v = None
            attn_out = self.mla_attn(
                q_full,
                kv_latent,
                k_pe_head,
                output_shape=(num_tokens, self.num_local_heads * self.v_head_dim),
            )
            attn_out = attn_out.view(num_tokens, self.num_local_heads, self.v_head_dim)
        else:
            # SWA: materialize K/V via kv_b_proj, run standard GQA attention
            # through the diff-KV flash backend (asymmetric qk/v head_dim).
            kv_proj, _ = self.kv_b_proj(kv_latent)          # [T, local_kv_heads * (nope + v_head_dim)]
            kv_proj = kv_proj.view(
                num_tokens, self.num_local_kv_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_proj.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe_expanded = k_pe_head.expand(-1, self.num_local_kv_heads, -1)  # [T, kv_heads, qk_rope_head_dim]
            k_full = torch.cat([k_nope, k_pe_expanded], dim=-1)  # [T, local_kv_heads, head_dim]

            if _save_gdl:
                _gdl['q_full'] = q_full.detach().float().cpu()
                _gdl['k_full'] = k_full.detach().float().cpu()
                _gdl['v'] = v.detach().float().cpu()

            # Pad V to head_dim so flash-attn head size matches Q/K.
            # In production self._use_sdpa=False (env var unset), so this branch
            # is dead-code-eliminated by Dynamo and the SDPA import never runs.
            if self._use_sdpa and not torch.compiler.is_compiling():
                # Bypass vLLM paged attention — call SDPA directly like TT
                v_padded = F.pad(v, [0, self.head_dim - self.v_head_dim])
                q_4d = q_full.unsqueeze(0).transpose(1, 2)   # [1, heads, T, head_dim]
                k_4d = k_full.unsqueeze(0).transpose(1, 2)   # [1, kv_heads, T, head_dim]
                v_4d = v_padded.unsqueeze(0).transpose(1, 2)  # [1, kv_heads, T, head_dim]
                from torch.nn.attention import SDPBackend, sdpa_kernel
                _sdpa_backend = SDPBackend.MATH if self._sdpa_math else SDPBackend.FLASH_ATTENTION
                with sdpa_kernel([_sdpa_backend]):
                    attn_out = F.scaled_dot_product_attention(
                        q_4d, k_4d, v_4d, is_causal=True, enable_gqa=True
                    )
                attn_out = attn_out.squeeze(0).transpose(0, 1)  # [T, heads, head_dim]
                attn_out = attn_out[..., : self.v_head_dim]
            else:
                # head_size_v=v_head_dim is passed to Attention(), so V keeps its
                # native dim (128) and FA takes the supported (qk=192, v=128) path.
                v_flat = v.reshape(num_tokens, self.num_local_kv_heads * self.v_head_dim)
                q_flat = q_full.reshape(num_tokens, self.num_local_heads * self.head_dim)
                k_flat = k_full.reshape(num_tokens, self.num_local_kv_heads * self.head_dim)

                attn_out = self.attn(q_flat, k_flat, v_flat)
                attn_out = attn_out.view(num_tokens, self.num_local_heads, self.v_head_dim)

        # ── Per-layer trace (all layers) ──
        _trace_all = self._trace_all

        # Split into signal/noise groups: num_local_heads = (grouped_ratio+1) * local_noise_heads
        attn_groups = attn_out.view(
            num_tokens, self.local_noise_heads, self.grouped_ratio + 1, self.v_head_dim
        )
        attn1 = attn_groups[:, :, :self.grouped_ratio, :].reshape(
            num_tokens, self.local_signal_heads, self.v_head_dim
        )  # signal
        attn2 = attn_groups[:, :, self.grouped_ratio:, :].reshape(
            num_tokens, self.local_noise_heads, self.v_head_dim
        )  # noise
        attn2 = attn2.repeat_interleave(self.grouped_ratio, dim=1)  # [T, local_signal_heads, v_head_dim]

        # Differential combination: signal - sigmoid(lambda) * noise
        # lambda_vals is float32 (lambda_proj uses params_dtype=float32);
        # cast result back to activation dtype before combining.
        lambda_scale = torch.sigmoid(lambda_vals.float()).to(attn1.dtype).unsqueeze(-1)
        attn_result = attn1 - lambda_scale * attn2

        if gate_score is not None:
            attn_result = attn_result * torch.sigmoid(gate_score)

        if _save_gdl:
            _gdl['diff_attn_output'] = attn_result.detach().float().cpu()

        output, _ = self.o_proj(attn_result.reshape(num_tokens, self.local_signal_heads * self.v_head_dim))

        if _trace_all and not torch.compiler.is_compiling():
            _trace_path = f"/tmp/vllm_trace_L{self.layer_idx}.pt"
            _trace_blob = {
                'layer_idx': self.layer_idx,
                'q_full': q_full.detach().float().cpu(),
                'attn_out': attn_out.detach().float().cpu(),
                'diff_attn_out': attn_result.detach().float().cpu(),
                'wo_output': output.detach().float().cpu(),
                'is_mla_layer': self.is_mla_layer,
            }
            if not self.is_mla_layer:
                # MLA materializes K/V inside MLAAttention's absorb path and
                # does not expose them; only SWA has them as local tensors.
                _trace_blob['k_full'] = k_full.detach().float().cpu()
                _trace_blob['v'] = v.detach().float().cpu()
            torch.save(_trace_blob, _trace_path)

        if _save_gdl and not torch.compiler.is_compiling():
            _gdl['wo_output'] = output.detach().float().cpu()
            torch.save(_gdl, "/tmp/vllm_gdl_L0.pt")
            logger.info(f"Saved vLLM GDL L0 ({len(_gdl)} keys)")

        return output


# ---------------------------------------------------------------------------
# Dense MLP
# ---------------------------------------------------------------------------

class MotifMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        hidden_clamp: Optional[float] = None,
        polynorm_output_scale: float = 1.0,
        polynorm_sigmoid_weight: bool = True,
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.down_proj",
        )
        if hidden_act == "poly_norm":
            self.act_fn = PolyNormTorch(sigmoid_weight=polynorm_sigmoid_weight)
        else:
            from vllm.model_executor.layers.activation import get_act_fn
            self.act_fn = get_act_fn(hidden_act)
        self.hidden_clamp = hidden_clamp
        self.polynorm_output_scale = polynorm_output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        if (
            isinstance(self.act_fn, PolyNormTorch)
            and self.act_fn._tp_size == 1
            and self.hidden_clamp is not None
        ):
            # Compiled fast path: clamp + PolyNorm + (gate * up) + scale all
            # fused into one inductor region; sigmoid'd weight is cached on
            # the PolyNormTorch instance.
            gated = _poly_norm_mlp_compute(
                gate, up,
                self.act_fn._get_weight(),
                self.act_fn.bias,
                self.act_fn.eps,
                self.hidden_clamp,
                self.polynorm_output_scale,
            )
        else:
            if self.hidden_clamp is not None:
                gate = gate.clamp(-self.hidden_clamp, self.hidden_clamp)
                up = up.clamp(-self.hidden_clamp, self.hidden_clamp)
            if isinstance(self.act_fn, PolyNormTorch):
                # fp32 `poly * up` then single downcast (matches training).
                # Covers the TP>1 dense/shared path, which is the common case in
                # production (the fused TP=1 path above does the same).
                gated = self.act_fn.forward_mul(gate, up)
            else:
                gated = self.act_fn(gate) * up
            if self.polynorm_output_scale != 1.0:
                gated = gated * self.polynorm_output_scale
        out, _ = self.down_proj(gated)
        return out


# ---------------------------------------------------------------------------
# FusedMoE-based MoE (TP-only via IdentityAllReduce backend, or EP via all2all)
# ---------------------------------------------------------------------------

def _motif_gate_tf32_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # TF32 tensor-core GEMM for the fp32 router gate.
    # allow_tf32 is a process-global cublas flag; toggling it here is safe
    # because the op body only ever runs single-threaded (eager warmup /
    # cudagraph capture), and the captured kernel keeps the TF32 choice.
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return torch.nn.functional.linear(x, weight)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _motif_gate_tf32_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="motif_gate_tf32_gemm",
    op_func=_motif_gate_tf32_gemm,
    mutates_args=[],
    fake_impl=_motif_gate_tf32_gemm_fake,
)


class MotifMoEFused(nn.Module):
    """SharedFusedMoE-backed MoE.

    Single MoE path used for BOTH TP-only and EP-enabled serving. Routing
    (sigmoid/softmax + topk + e_score_correction_bias + renorm + scale) is
    delegated to ``FusedTopKBiasRouter`` (constructed inside ``SharedFusedMoE``
    from the constructor args). Per-expert PolyNorm activation is wired in
    via ``MotifMoEMethod`` which selects ``MotifTritonExperts`` as the GEMM
    impl. Shared-expert single-allreduce fusion is handled by ``SharedFusedMoE``.

    Parallelism modes:
      * ``--enable-expert-parallel`` set → vLLM's standard EP topology
        (DP attention + EP all2all dispatch/combine via deepep / naive /
        allgather_reducescatter / flashinfer_all2allv backend).
      * ``--enable-expert-parallel`` unset (TP-only) → experts are sharded
        across TP world group (``ep_size = tp_size`` is forced internally
        during ``SharedFusedMoE`` construction). Communication is a single
        ``tensor_model_parallel_all_reduce`` of the partial output buffer,
        delivered via ``MoEPrepareAndFinalizeIdentityAllReduce``.
    """

    def __init__(
        self,
        config: MotifConfig,
        polynorm_output_scale: float = 1.0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_clamp = getattr(config, "hidden_clamp", 1000)
        polynorm_sigmoid_weight = getattr(config, "polynorm_sigmoid_weight", True)
        # Routed-expert PolyNorm bias clamp. Training clamps the bias to
        # [-c, c] inside GroupedExpertsPolyNorm.forward every step; the bias is
        # frozen at inference, so clamping the loaded act_fn_bias param once is
        # numerically identical and covers all expert backends (Triton / MXFP8 /
        # DeepGEMM) since they share the same parameter. None = no clamp.
        self.bias_clamp = getattr(config, "polynorm_bias_clamp", None)

        # MXFP8 auto-conversion: pass quant_config=None to SharedFusedMoE so
        # UnquantizedFusedMoEMethod allocates bf16 weights for the standard
        # loader; MotifMxfp8MoEMethod (installed below) quantizes them in-place
        # to MXFP8 in process_weights_after_loading. Shared experts + linears
        # stay bf16 (the linear MXFP8 path needs a serialized checkpoint).
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptBlockFp8Config,
            ModelOptMxFp8Config,
            ModelOptNvFp4DynamicConfig,
        )
        # ModelOptBlockFp8Config / ModelOptNvFp4DynamicConfig subclass
        # ModelOptMxFp8Config, so _is_mxfp8 is True for all three — that gates
        # the shared "load bf16, quantize at load" path (correct for all).
        # _is_blockfp8 / _is_nvfp4 then pick the DeepGEMM 1x128 / CUTLASS NVFP4
        # method over the CUTLASS MXFP8 one.
        self._is_mxfp8 = isinstance(quant_config, ModelOptMxFp8Config)
        self._is_blockfp8 = isinstance(quant_config, ModelOptBlockFp8Config)
        self._is_nvfp4 = isinstance(quant_config, ModelOptNvFp4DynamicConfig)
        # Direct load: the checkpoint already carries packed NVFP4 expert
        # tensors (tools/motif_nvfp4_quantize_ckpt.py); skip load-time
        # quantization and load them as-is.
        self._nvfp4_direct = self._is_nvfp4 and getattr(
            quant_config, "direct_load", False
        )
        experts_quant_config = None if self._is_mxfp8 else quant_config
        shared_quant_config = None if self._is_mxfp8 else quant_config

        # Capture the user-requested EP mode before any internal toggling.
        from vllm.config import get_current_vllm_config
        _vllm_config = get_current_vllm_config()
        _user_enable_ep = _vllm_config.parallel_config.enable_expert_parallel
        # In TP-only mode (no --enable-expert-parallel), force the FusedMoE
        # parallel-config to treat experts as sharded across the TP world
        # (ep_size = tp_size). The user-facing flag remains False, so
        # attention path stays TP. We restore the flag after SharedFusedMoE
        # construction to avoid leaking the override to other layers.
        self._use_identity_allreduce = not _user_enable_ep

        # fp32 gate weights; the GEMM itself runs as TF32 on tensor cores via
        # the motif_gate_tf32_gemm custom op (fp32 storage/accumulate, 10-bit
        # mantissa multiply). With sigmoid scoring + e_score_correction_bias,
        # bf16 gate weights shift borderline top-k decisions (4-5% top-8 set
        # flips measured) and degrade accuracy; TF32 keeps flips at ~0.01-0.04%
        # while being ~2.5x faster than the fp32 SIMT sgemm at decode M.
        # We do NOT pass gate=self.gate to SharedFusedMoE below, which means
        # FusedMoE's runner does not call the gate internally — only our
        # explicit cast path in forward runs.
        self.gate_dtype = torch.float32
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=self.gate_dtype,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if config.load_balance_coeff is not None:
            # vLLM's CUDA `topk_softmax`/`topk_sigmoid` kernels enforce fp32 bias
            # (csrc/moe/topk_softmax_kernels.cu: TORCH_CHECK bias == Float).
            # `gating_output` is template-dispatched, so its dtype is free.
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.zeros(config.num_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.gate.e_score_correction_bias = None

        # Shared expert (SharedFusedMoE folds its output into the routed
        # all-reduce, so disable the inner reduce on its down_proj).
        if config.num_shared_experts > 0:
            shared = MotifMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=shared_quant_config,
                prefix=f"{prefix}.shared_experts",
                hidden_clamp=self.hidden_clamp,
                polynorm_output_scale=polynorm_output_scale,
                polynorm_sigmoid_weight=polynorm_sigmoid_weight,
            )
            shared.down_proj.reduce_results = False
            self.shared_experts = shared
        else:
            self.shared_experts = None

        # NOTE: do NOT pass gate=self.gate. The runner's overlap path
        # (default_moe_runner.py:617) calls gate without dtype cast; we want
        # only our explicit fp32 cast in MotifMoEFused.forward to invoke the
        # gate. Loses gate-dispatch overlap (small perf), gains correctness.
        #
        # In TP-only mode, temporarily flip enable_expert_parallel=True so
        # FusedMoEParallelConfig.make() returns ep_size=tp_size (sharded
        # experts) instead of ep_size=1 (replicated). The flag is restored
        # immediately after construction. The IdentityAllReduce backend
        # (selected via use_identity_allreduce below) then takes over the
        # collective-communication semantics.
        if self._use_identity_allreduce:
            _vllm_config.parallel_config.enable_expert_parallel = True
        try:
            self.experts = SharedFusedMoE(
                shared_experts=self.shared_experts,
                num_experts=config.num_experts,
                top_k=config.experts_top_k,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                renormalize=config.route_norm,
                scoring_func=config.score_func,
                routed_scaling_factor=config.route_scale,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                router_logits_dtype=torch.float32,
                quant_config=experts_quant_config,
                prefix=f"{prefix}.experts",
                apply_router_weight_on_input=False,
                activation="silu",  # placeholder; MotifMoEMethod replaces activation
            )
        finally:
            if self._use_identity_allreduce:
                _vllm_config.parallel_config.enable_expert_parallel = _user_enable_ep

        # Per-expert PolyNorm parameters live on the FusedMoE layer; they are
        # sliced into local-experts via custom load_weights logic.
        local_E = self.experts.local_num_experts
        self.experts.act_fn_weight = nn.Parameter(
            torch.empty(local_E, 3, dtype=torch.float32)
        )
        self.experts.act_fn_bias = nn.Parameter(
            torch.empty(local_E, 1, dtype=torch.float32)
        )

        # Inject motif's GEMM impl: replace quant_method with MotifMoEMethod.
        # Also overwrite `base_quant_method` — `FusedMoE.__init__` captures it as
        # the original UnquantizedFusedMoEMethod and `maybe_init_modular_kernel`
        # later calls `base_quant_method.select_gemm_impl(...)`. Without this
        # update the original (whose `moe_quant_config` was never populated by
        # `process_weights_after_loading`) is used and assertion fails.
        # --quantization modelopt_blockfp8 selects the block-wise FP8 (128x128
        # weight / 1x128 activation) DeepGEMM grouped-GEMM path on SM90 and
        # SM100 alike (falls back to vllm.third_party.deep_gemm when the
        # standalone deep_gemm package is absent). modelopt_mxfp8 keeps the
        # original CUTLASS MXFP8 (1x32) logic.
        if self._is_nvfp4:
            # --quantization modelopt_nvfp4 selects the CUTLASS NVFP4 path
            # (E2M1 + 1x16 E4M3 blockscale, SM100+ only).
            from vllm.model_executor.layers.fused_moe.motif_nvfp4_experts import (
                MotifNvfp4MoEMethod,
            )
            motif_method = MotifNvfp4MoEMethod(
                moe=self.experts.moe_config,
                poly_norm_weight=self.experts.act_fn_weight,
                poly_norm_bias=self.experts.act_fn_bias,
                hidden_clamp=self.hidden_clamp,
                polynorm_output_scale=polynorm_output_scale,
                polynorm_sigmoid_weight=polynorm_sigmoid_weight,
                use_identity_allreduce=self._use_identity_allreduce,
                direct_load=self._nvfp4_direct,
            )
            if self._nvfp4_direct:
                motif_method.convert_layer_for_direct_load(self.experts)
        elif self._is_blockfp8:
            # DeepGEMM 1x128 grouped GEMM — the only block-FP8 path
            # (SM90 and SM100; validated on H200, PR #63).
            from vllm.model_executor.layers.fused_moe.motif_deepgemm_experts import (  # noqa: E501
                MotifDeepGemmMoEMethod as _blockfp8_method_cls,
            )
            logger.info(
                "modelopt_blockfp8 MoE method: {}", _blockfp8_method_cls.__name__
            )
            motif_method = _blockfp8_method_cls(
                moe=self.experts.moe_config,
                poly_norm_weight=self.experts.act_fn_weight,
                poly_norm_bias=self.experts.act_fn_bias,
                hidden_clamp=self.hidden_clamp,
                polynorm_output_scale=polynorm_output_scale,
                polynorm_sigmoid_weight=polynorm_sigmoid_weight,
                use_identity_allreduce=self._use_identity_allreduce,
            )
        elif self._is_mxfp8:
            from vllm.model_executor.layers.fused_moe.motif_mxfp8_experts import (
                MotifMxfp8MoEMethod,
            )
            motif_method = MotifMxfp8MoEMethod(
                moe=self.experts.moe_config,
                poly_norm_weight=self.experts.act_fn_weight,
                poly_norm_bias=self.experts.act_fn_bias,
                hidden_clamp=self.hidden_clamp,
                polynorm_output_scale=polynorm_output_scale,
                polynorm_sigmoid_weight=polynorm_sigmoid_weight,
                use_identity_allreduce=self._use_identity_allreduce,
            )
        else:
            from vllm.model_executor.layers.fused_moe.motif_experts import (
                MotifMoEMethod,
            )
            motif_method = MotifMoEMethod(
                moe=self.experts.moe_config,
                poly_norm_weight=self.experts.act_fn_weight,
                poly_norm_bias=self.experts.act_fn_bias,
                hidden_clamp=self.hidden_clamp,
                polynorm_output_scale=polynorm_output_scale,
                polynorm_sigmoid_weight=polynorm_sigmoid_weight,
                use_identity_allreduce=self._use_identity_allreduce,
            )
        self.experts._replace_quant_method(motif_method)
        self.experts.base_quant_method = motif_method

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Mirrors deepseek_v2: only the internal-router path may use
        # `router_logits=hidden_states` as a placeholder (gate is invoked
        # inside FusedMoE). All other backends require pre-computed
        # router_logits — without this, garbage indices are passed to DeepEP
        # `get_dispatch_layout`, surfacing as a CUDA launch failure.
        # v0.20.2: FusedMoE folds shared_experts internally and returns a
        # single tensor (shared + routed already combined).
        if self.experts.is_internal_router:
            return self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
        # fp32 storage, TF32 tensor-core multiply (see __init__ comment).
        router_logits = torch.ops.vllm.motif_gate_tf32_gemm(
            hidden_states.to(torch.float32), self.gate.weight
        )
        return self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class MotifDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        layer_idx: Optional[int] = None,
        config_override: Optional["MotifConfig"] = None,
    ):
        super().__init__()
        # MTP predictor blocks pass an explicit layer_idx and a config_override
        # (a SWA/dense clone) so the block is built independently of the base
        # model's hf_config and prefix-derived index.
        config: MotifConfig = config_override or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if layer_idx is not None:
            self.layer_idx = layer_idx
        else:
            self.layer_idx = int(prefix.split(".")[-1])
        self.mhc_enabled = config.mhc_enabled

        self.self_attn = MotifGDLAttention(
            config=config,
            layer_idx=self.layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        # MoE vs dense MLP
        n_dense_first = config.n_dense_first_layers
        step = config.interleave_moe_layer_step
        self.moe_enabled = (
            self.layer_idx >= n_dense_first
            and step != 0
            and (self.layer_idx + 1) % step == 0
        )
        _pn_per_layer = getattr(config, "polynorm_output_scale_per_layer", None) or {}
        # JSON deserializes dict keys as strings; look up both int and str forms.
        _pn_key = _pn_per_layer.get(self.layer_idx) or _pn_per_layer.get(str(self.layer_idx))
        if _pn_key is not None:
            polynorm_output_scale = float(_pn_key)
        else:
            polynorm_output_scale = float(getattr(config, "polynorm_output_scale", 1.0))

        if self.moe_enabled:
            # Single MoE path: MotifMoEFused for both TP-only and EP-enabled
            # serving. Backend selection (IdentityAllReduce vs all2all) happens
            # inside MotifMoEMethod based on parallel_config.enable_expert_parallel.
            self.moe = MotifMoEFused(
                config,
                polynorm_output_scale=polynorm_output_scale,
                quant_config=quant_config,
                prefix=f"{prefix}.moe",
            )
        else:
            self.mlp = MotifMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                polynorm_output_scale=polynorm_output_scale,
                polynorm_sigmoid_weight=getattr(config, "polynorm_sigmoid_weight", True),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._mhc_use_tilelang = False
        if self.mhc_enabled:
            # Training uses (1 + alpha)*sigmoid with alpha decaying to mhc_h_post_alpha_end.
            # At inference the schedule is finished, so coeff = 1 + alpha_end.
            _h_post_coeff = 1.0 + float(getattr(config, "mhc_h_post_alpha_end", 0.0))
            # DeepSeek-V4 tilelang/DeepGEMM MHC kernels on by default; opt out
            # (fall back to the Triton MHC path) via MOTIF_MHC_TILELANG=0.
            _mhc_tilelang = os.environ.get("MOTIF_MHC_TILELANG", "1") not in (
                "0", "", "false", "False",
            )
            self._mhc_use_tilelang = _mhc_tilelang
            self.mhc_attn = MotifMHCLayer(
                expansion_rate=config.mhc_expansion_rate,
                num_dim=config.hidden_size,
                identity_init=config.mhc_identity_init,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                h_post_coeff=_h_post_coeff,
                use_tilelang=_mhc_tilelang,
            )
            self.mhc_ffn = MotifMHCLayer(
                expansion_rate=config.mhc_expansion_rate,
                num_dim=config.hidden_size,
                identity_init=config.mhc_identity_init,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                h_post_coeff=_h_post_coeff,
                use_tilelang=_mhc_tilelang,
            )

        # Debug env-var snapshots (init-time so forward stays compile-safe).
        self._trace_all = bool(os.environ.get("MOTIF_TRACE_ALL"))
        self._mhc_module_test = bool(os.environ.get("MOTIF_MODULE_TEST"))

    def _run_mlp_or_moe(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.moe_enabled:
            return self.moe(hidden_states)
        return self.mlp(hidden_states)

    def forward(self, positions, hidden_states, residual):
        L = self.layer_idx
        if self.mhc_enabled:
            hidden_states = self._forward_mhc(positions, hidden_states)
            return hidden_states, None
        else:
            # Save layer input (residual stream) for tracing.
            # Guarded so torch.compile (which only ever sees the non-MHC branch)
            # does not graph-break on the file write.
            if self._trace_all and not torch.compiler.is_compiling():
                _layer_input = (residual if residual is not None else hidden_states).detach().float().cpu()
                _trace_path = f"/tmp/vllm_trace_L{L}_input.pt"
                torch.save({'layer_idx': L, 'layer_input': _layer_input}, _trace_path)

            _stat(f"L{L:02d}.pre_attn_norm            ", hidden_states if residual is None else residual)
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            _stat(f"L{L:02d}.attn_norm_out            ", hidden_states)

            hidden_states = self.self_attn(positions, hidden_states)

            _stat(f"L{L:02d}.pre_ffn_norm             ", residual)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            _stat(f"L{L:02d}.ffn_norm_out             ", hidden_states)

            hidden_states = self._run_mlp_or_moe(hidden_states)
            _stat(f"L{L:02d}.output                   ", hidden_states)
            return hidden_states, residual

    def _forward_mhc(self, positions, x):
        if not torch.compiler.is_compiling() and x.shape[0] == 0:
            # Compile-time dead-coded; only fires for dummy profile runs.
            return x
        if self._mhc_use_tilelang:
            return self._forward_mhc_tilelang(positions, x)
        L = self.layer_idx
        x = x.unsqueeze(0)

        # Snapshot-based debug flags. is_compiling() check makes the entire
        # debug pathway invisible to Dynamo's tracer (constant False).
        _trace_all = self._trace_all and not torch.compiler.is_compiling()
        _module_test = self._mhc_module_test and not torch.compiler.is_compiling()

        # Same-input module test: load TT's saved inputs and run modules on them.
        if _module_test:
            _inp_path = f"/tmp/same_input_L{L}.pt"
            if os.path.exists(_inp_path):
                _inp = torch.load(_inp_path, weights_only=False)
                _x4d  = _inp['x4d'].to(x.device).to(x.dtype)
                _fi   = _inp['ffn_in'].to(x.device).to(x.dtype)
                _hp, _hpost, _hr = self.mhc_attn(_x4d)
                _x_red = MotifMHCLayer.apply_h_pre(_x4d, _hp).squeeze(0)
                _hp_f, _hpost_f, _hr_f = self.mhc_ffn(_x4d)
                _h_red = MotifMHCLayer.apply_h_pre(_x4d, _hp_f).squeeze(0)
                _ffn_out = self._run_mlp_or_moe(_fi)
                torch.save({
                    'layer_idx': L,
                    'is_moe': self.moe_enabled,
                    'mhc_attn_h_pre':  _hp.detach().float().cpu(),
                    'mhc_attn_h_post': _hpost.detach().float().cpu(),
                    'mhc_attn_pre_out': _x_red.detach().float().cpu(),
                    'mhc_ffn_h_pre':   _hp_f.detach().float().cpu(),
                    'mhc_ffn_h_post':  _hpost_f.detach().float().cpu(),
                    'mhc_ffn_pre_out': _h_red.detach().float().cpu(),
                    'ffn_out': _ffn_out.detach().float().cpu(),
                }, f"/tmp/vllm_same_out_L{L}.pt")

        # Per-layer activation tracing. _module_trace stays None when compiled
        # or when MOTIF_TRACE_ALL unset, so subsequent `is not None` checks
        # are constant-False at trace time → dead-coded.
        _module_trace = None
        if _trace_all:
            _layer_input = x.mean(dim=2).squeeze(0).detach().float().cpu() if x.dim() == 4 else x.squeeze(0).detach().float().cpu()
            torch.save({'layer_idx': L, 'layer_input': _layer_input}, f"/tmp/vllm_trace_L{L}_input.pt")
            _module_trace = {'layer_idx': L, 'mhc_attn_pre_in': _layer_input}

        h_pre, h_post, h_res = self.mhc_attn(x)
        x_reduced = MotifMHCLayer.apply_h_pre(x, h_pre).squeeze(0)
        if _module_trace is not None:
            _module_trace['mhc_attn_pre_out'] = x_reduced.detach().float().cpu()

        attn_in = self.input_layernorm(x_reduced)
        attn_out = self.self_attn(positions, attn_in)

        # MHC post: h_res @ x + h_post * attn_out
        if _HAS_TT_MHC_KERNELS and x.is_cuda:
            h = _tt_mhc_post_fused(h_res.contiguous(), x.contiguous(),
                                   h_post.contiguous(), attn_out.unsqueeze(0).contiguous())
        else:
            _attn_res_fp32 = torch.einsum("bsij,bsjd->bsid", h_res, x.float())
            _attn_post_fp32 = h_post.unsqueeze(-1) * attn_out.float().unsqueeze(0).unsqueeze(2)
            h = (_attn_res_fp32 + _attn_post_fp32).to(x.dtype)
        if _module_trace is not None:
            _module_trace['mhc_attn_post_out'] = h.mean(dim=2).squeeze(0).detach().float().cpu() if h.dim() == 4 else h.squeeze(0).detach().float().cpu()

        h_pre_f, h_post_f, h_res_f = self.mhc_ffn(h)
        h_reduced = MotifMHCLayer.apply_h_pre(h, h_pre_f).squeeze(0)
        if _module_trace is not None:
            _module_trace['mhc_ffn_pre_out'] = h_reduced.detach().float().cpu()

        ffn_in = self.post_attention_layernorm(h_reduced)
        if _module_trace is not None:
            _module_trace['ffn_in'] = ffn_in.detach().float().cpu()

        ffn_out = self._run_mlp_or_moe(ffn_in)
        if _module_trace is not None:
            _module_trace['ffn_out'] = ffn_out.detach().float().cpu()
            _module_trace['is_moe'] = self.moe_enabled

        # MHC FFN post: h_res_f @ h + h_post_f * ffn_out
        if _HAS_TT_MHC_KERNELS and h.is_cuda:
            out = _tt_mhc_post_fused(h_res_f.contiguous(), h.contiguous(),
                                     h_post_f.contiguous(), ffn_out.unsqueeze(0).contiguous())
        else:
            _ffn_res_fp32 = torch.einsum("bsij,bsjd->bsid", h_res_f, h.float())
            _ffn_post_fp32 = h_post_f.unsqueeze(-1) * ffn_out.float().unsqueeze(0).unsqueeze(2)
            out = (_ffn_res_fp32 + _ffn_post_fp32).to(h.dtype)
        if _module_trace is not None:
            _module_trace['mhc_ffn_post_out'] = out.mean(dim=2).squeeze(0).detach().float().cpu() if out.dim() == 4 else out.squeeze(0).detach().float().cpu()
            torch.save(_module_trace, f"/tmp/vllm_module_L{L}.pt")

        return out.squeeze(0)

    def _forward_mhc_tilelang(self, positions, x):
        """Tilelang/DeepGEMM MHC path (default on; opt out via MOTIF_MHC_TILELANG=0).

        x: (T, E, D) bf16 residual stream -> (T, E, D). Same structure as the
        Triton `_forward_mhc`, with the fused torch.ops.vllm.mhc_pre / mhc_post
        ops replacing proj + RMS + sigmoid + sinkhorn + apply_h_pre +
        mhc_post_fused (parity: mhc_tilelang_parity.py).
        """
        x = x.contiguous()
        # Attention sub-block.
        post_mix, comb_mix, layer_input = self.mhc_attn.mhc_pre_tilelang(x)
        attn_in = self.input_layernorm(layer_input)
        attn_out = self.self_attn(positions, attn_in)
        h = torch.ops.vllm.mhc_post(attn_out.contiguous(), x, post_mix, comb_mix)

        # FFN / MoE sub-block.
        post_mix_f, comb_mix_f, layer_input_f = self.mhc_ffn.mhc_pre_tilelang(h)
        ffn_in = self.post_attention_layernorm(layer_input_f)
        ffn_out = self._run_mlp_or_moe(ffn_in)
        return torch.ops.vllm.mhc_post(ffn_out.contiguous(), h, post_mix_f, comb_mix_f)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Compile is enabled for both MHC and non-MHC paths:
#   - Data-dependent T==0 guards are wrapped under not torch.compiler.is_compiling()
#     (dead-coded at trace time).
#   - MHC triton kernels (sinkhorn_fused / res_triton / mhc_post_fused) are
#     registered as torch.library.custom_op so Dynamo treats them as opaque
#     ops instead of graph-breaking on raw triton.jit calls.
#   - Debug env-vars are snapshotted at __init__ and combined with
#     not torch.compiler.is_compiling() so all save/trace branches are
#     constant-False at trace time.
@support_torch_compile
class MotifModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MotifConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.mhc_enabled = config.mhc_enabled
        self.mhc_expansion_rate = config.mhc_expansion_rate

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MotifDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # Debug env-var snapshot (init-time, no forward-path env reads).
        self._save_layers = bool(os.environ.get("MOTIF_SAVE_LAYERS"))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            # MHC: expand [T, D] -> [T, E, D]
            if self.mhc_enabled:
                hidden_states = hidden_states.unsqueeze(1).expand(
                    -1, self.mhc_expansion_rate, -1
                ).contiguous()
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Init-time snapshot — env-var reads in forward break torch.compile.
        _save_layers = self._save_layers
        _layer_snapshots: dict = {}

        # Plain integer indexing instead of itertools.islice — Dynamo traces
        # range(int, int) cleanly but the islice constructor introduces a
        # graph break.
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            hidden_states, residual = layer(positions, hidden_states, residual)
            if _save_layers and not torch.compiler.is_compiling():
                snap = hidden_states.mean(dim=1) if (self.mhc_enabled and hidden_states.dim() == 3) else hidden_states
                _layer_snapshots[layer_idx] = snap.detach().float().cpu()

        if _save_layers and not torch.compiler.is_compiling():
            torch.save(_layer_snapshots, "/tmp/vllm_layer_states.pt")
            logger.info("Saved vLLM layer states to /tmp/vllm_layer_states.pt")

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        if self.mhc_enabled:
            hidden_states = hidden_states.mean(dim=1)

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# CausalLM
# ---------------------------------------------------------------------------

class MotifForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MotifConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = MotifModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self._mhc_fused_ready = False

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        self._ensure_mhc_fused()
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def _ensure_mhc_fused(self) -> None:
        """Build tilelang MHC fused params once, lazily, in the eager outer
        forward (before MotifModel's compiled region / cudagraph capture).
        Runs after weight loading, and also covers --load-format dummy, which
        skips model.load_weights().

        RL refit contract: NeMo-RL's tilelang_mhc.refit_fused_params keys off
        ``_mhc_fused_ready`` (and per-layer ``_fused_built``) to refresh the
        fused buffers in place after a weight push — keep the flag semantics
        and name stable."""
        if self._mhc_fused_ready:
            return
        for m in self.modules():
            if isinstance(m, MotifMHCLayer) and getattr(m, "_use_tilelang", False):
                m.build_fused_params()
        self._mhc_fused_ready = True


    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def _find_moe_for_param(self, name: str):
        """Walk the dotted path; return the first MotifMoEFused found."""
        if "moe" not in name:
            return None
        obj = self
        for p in name.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                return None
            if isinstance(obj, MotifMoEFused):
                return obj
        return None

    def _load_fused_moe_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        moe_module: "MotifMoEFused",
        params_dict: dict,
        loaded_params: set,
    ) -> bool:
        """FusedMoE-path expert weight loading. Returns True if consumed."""
        fused_layer = moe_module.experts

        # NVFP4 direct load (tools/motif_nvfp4_quantize_ckpt.py): per-expert
        # global scales [E] fp32. Custom expert_map slicing — FusedMoE's
        # per-tensor scale loader expects a different param layout. Checked
        # before the *_weight_scale / plain weight fragments (substrings).
        for ckpt_frag, vllm_frag in (
            ("moe.experts.gate_up_proj_weight_scale_2",
             "moe.experts.w13_weight_scale_2"),
            ("moe.experts.down_proj_weight_scale_2",
             "moe.experts.w2_weight_scale_2"),
        ):
            if ckpt_frag not in name:
                continue
            target = name.replace(ckpt_frag, vllm_frag)
            param = params_dict.get(target)
            if param is None:
                raise ValueError(
                    f"{name}: checkpoint carries NVFP4 expert scales but the "
                    "direct-load NVFP4 config is not active — serve this "
                    "checkpoint with its own config.json (its "
                    "quantization_config selects modelopt_nvfp4 direct load)"
                )
            assert loaded_weight.dim() == 1, (
                f"{name} must be [E], got {loaded_weight.shape}"
            )
            expert_map = getattr(fused_layer, "_expert_map", None)
            if expert_map is None:
                param.data.copy_(loaded_weight.to(param.dtype))
            else:
                for ge in range(loaded_weight.shape[0]):
                    le = int(expert_map[ge].item())
                    if le < 0:
                        continue
                    param.data[le] = loaded_weight[ge].to(param.dtype)
            loaded_params.add(target)
            return True

        # NVFP4 direct load: linear-layout E4M3 blockscales, loaded through
        # FusedMoE's BLOCK group-scale path (same sharding as the weights).
        for ckpt_frag, vllm_frag, is_w13 in (
            ("moe.experts.gate_up_proj_weight_scale",
             "moe.experts.w13_weight_scale", True),
            ("moe.experts.down_proj_weight_scale",
             "moe.experts.w2_weight_scale", False),
        ):
            if ckpt_frag not in name:
                continue
            target = name.replace(ckpt_frag, vllm_frag)
            param = params_dict.get(target)
            if param is None:
                raise ValueError(
                    f"{name}: checkpoint carries NVFP4 expert scales but the "
                    "direct-load NVFP4 config is not active — serve this "
                    "checkpoint with its own config.json (its "
                    "quantization_config selects modelopt_nvfp4 direct load)"
                )
            assert loaded_weight.dim() == 3, (
                f"{name} must be [E, X, K/16], got {loaded_weight.shape}"
            )
            for ge in range(loaded_weight.shape[0]):
                if is_w13:
                    nb = loaded_weight.shape[1]
                    assert nb % 2 == 0
                    param.weight_loader(
                        param, loaded_weight[ge, : nb // 2, :], target,
                        "w1", ge,
                    )
                    param.weight_loader(
                        param, loaded_weight[ge, nb // 2 :, :], target,
                        "w3", ge,
                    )
                else:
                    param.weight_loader(
                        param, loaded_weight[ge], target, "w2", ge
                    )
            loaded_params.add(target)
            return True

        # gate_up_proj [E, 2*I, H] → w13_weight, split into w1 and w3 halves.
        # (bf16 checkpoint: [E, 2*I, H] bf16; NVFP4 direct load: packed uint8
        # [E, 2*I, H/2] — the FusedMoE loader shards on dim 0 of the
        # per-expert slice, so the packed last dim passes through whole.)
        if "moe.experts.gate_up_proj" in name:
            target = name.replace(
                "moe.experts.gate_up_proj", "moe.experts.w13_weight"
            )
            param = params_dict.get(target)
            if param is None:
                return True
            if (loaded_weight.dtype == torch.uint8) != (
                param.dtype == torch.uint8
            ):
                raise ValueError(
                    f"{name}: checkpoint dtype {loaded_weight.dtype} vs "
                    f"param {param.dtype} — a packed NVFP4 checkpoint "
                    "requires the direct-load config (keep the "
                    "quantization_config written into its config.json); a "
                    "bf16 checkpoint uses --quantization modelopt_nvfp4 "
                    "dynamic quantization"
                )
            assert loaded_weight.dim() == 3, (
                f"gate_up_proj must be 3D [E, 2*I, H], got {loaded_weight.shape}"
            )
            E_ckpt, two_I, _ = loaded_weight.shape
            assert two_I % 2 == 0
            I = two_I // 2
            for ge in range(E_ckpt):
                param.weight_loader(
                    param, loaded_weight[ge, :I, :], target, "w1", ge
                )
                param.weight_loader(
                    param, loaded_weight[ge, I:, :], target, "w3", ge
                )
            loaded_params.add(target)
            return True

        # down_proj [E, H, I] → w2_weight per expert. (NVFP4 direct load:
        # packed uint8 [E, H, I/2]; with sharded experts each expert stays
        # full-width so the packed dim never splits.)
        if "moe.experts.down_proj" in name:
            target = name.replace(
                "moe.experts.down_proj", "moe.experts.w2_weight"
            )
            param = params_dict.get(target)
            if param is None:
                return True
            if (loaded_weight.dtype == torch.uint8) != (
                param.dtype == torch.uint8
            ):
                raise ValueError(
                    f"{name}: checkpoint dtype {loaded_weight.dtype} vs "
                    f"param {param.dtype} — a packed NVFP4 checkpoint "
                    "requires the direct-load config (keep the "
                    "quantization_config written into its config.json); a "
                    "bf16 checkpoint uses --quantization modelopt_nvfp4 "
                    "dynamic quantization"
                )
            E_ckpt = loaded_weight.shape[0]
            for ge in range(E_ckpt):
                param.weight_loader(
                    param, loaded_weight[ge], target, "w2", ge
                )
            loaded_params.add(target)
            return True

        # Per-expert PolyNorm params: act_fn.{weight,bias} [E, ...] sliced via
        # FusedMoE's expert_map (custom; FusedMoE weight_loader doesn't apply).
        for ckpt_frag, vllm_frag in (
            ("moe.experts.act_fn.weight", "moe.experts.act_fn_weight"),
            ("moe.experts.act_fn.bias", "moe.experts.act_fn_bias"),
        ):
            if ckpt_frag in name:
                target = name.replace(ckpt_frag, vllm_frag)
                param = params_dict.get(target)
                if param is None:
                    return True
                expert_map = getattr(fused_layer, "_expert_map", None)
                if expert_map is None:
                    # EP=1: all experts local, copy directly.
                    param.data.copy_(loaded_weight.to(param.dtype))
                else:
                    for ge in range(loaded_weight.shape[0]):
                        le = int(expert_map[ge].item())
                        if le < 0:
                            continue
                        param.data[le].copy_(
                            loaded_weight[ge].to(param.dtype)
                        )
                # Clamp the routed-expert PolyNorm bias to match training's
                # GroupedExpertsPolyNorm.forward (bias.clamp(-c, c)). Idempotent,
                # so clamping the whole param after each incremental EP copy is
                # safe. Done at load time because the bias is frozen at inference
                # and all expert backends read this same parameter.
                if (
                    vllm_frag == "moe.experts.act_fn_bias"
                    and moe_module.bias_clamp is not None
                ):
                    c = float(moe_module.bias_clamp)
                    param.data.clamp_(-c, c)
                loaded_params.add(target)
                return True

        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Common renames (attention + gate). MotifMoEFused redirects expert
        # weights through _load_fused_moe_expert_weight below.
        attn_rename = {
            "self_attn.wq_a": "self_attn.q_a_proj",
            "self_attn.q_norm": "self_attn.q_a_layernorm",
            "self_attn.wq_b.": "self_attn.q_b_proj.",
            "self_attn.wq_b_gate": "self_attn.q_b_gate",
            "self_attn.wkv_a": "self_attn.kv_a_proj_with_mqa",
            "self_attn.kv_norm": "self_attn.kv_a_layernorm",
            "self_attn.wkv_b": "self_attn.kv_b_proj",
            "self_attn.lambda_proj": "self_attn.lambda_proj",
            "self_attn.wo": "self_attn.o_proj",
            "moe.router.gate": "moe.gate",
            "moe.expert_bias": "moe.e_score_correction_bias",
        }

        # MHC: ckpt has 3 separate proj_pre/post/res; we pack them into a single
        # proj_merged via MergedColumnParallelLinear's per-shard weight_loader.
        # Order matches output_sizes=[E, E, E*E] in MotifMHCLayer.__init__.
        mhc_proj_shards = [
            ("proj_pre.weight", 0),
            ("proj_post.weight", 1),
            ("proj_res.weight", 2),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Apply renaming
            for ckpt_frag, vllm_frag in attn_rename.items():
                if ckpt_frag in name:
                    name = name.replace(ckpt_frag, vllm_frag)
                    break

            # MHC merged-projection routing.
            mhc_handled = False
            for ckpt_frag, shard_id in mhc_proj_shards:
                if name.endswith(ckpt_frag) and (".mhc_attn." in name or ".mhc_ffn." in name):
                    merged_name = name[: -len(ckpt_frag)] + "proj_merged.weight"
                    if merged_name in params_dict:
                        param = params_dict[merged_name]
                        param.weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(merged_name)
                    mhc_handled = True
                    break
            if mhc_handled:
                continue

            # Find the parent MotifMoEFused module if any.
            moe_module = self._find_moe_for_param(name)

            # MotifMoEFused-specific redirects and expert weight handling.
            if isinstance(moe_module, MotifMoEFused):
                # e_score_correction_bias lives on the gate, not on the moe.
                if name.endswith(".moe.e_score_correction_bias"):
                    name = name.replace(
                        ".moe.e_score_correction_bias",
                        ".moe.gate.e_score_correction_bias",
                    )
                # Expert-weight loading routes through FusedMoE's weight_loader.
                if self._load_fused_moe_expert_weight(
                    name, loaded_weight, moe_module,
                    params_dict, loaded_params,
                ):
                    continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            if name not in params_dict:
                continue

            # MLA layers size kv_b_proj as [num_heads*(P+V), Lkv] so the MLA
            # backend can absorb W_UK/W_UV per query head, but checkpoints store
            # the GQA shape [num_kv_heads*(P+V), Lkv]. Replicate each kv-head
            # block group_size=num_heads/num_kv_heads times so query heads in the
            # same GQA group share the same materialized K/V (mathematically
            # equivalent to GQA). SWA-mode layers keep the original shape.
            if name.endswith(".self_attn.kv_b_proj.weight"):
                attn_module = None
                try:
                    layer_idx = extract_layer_index(name)
                    attn_module = self.model.layers[layer_idx].self_attn
                except (AssertionError, IndexError, AttributeError):
                    pass
                if (
                    attn_module is not None
                    and getattr(attn_module, "is_mla_layer", False)
                    and attn_module.num_heads != attn_module.num_kv_heads
                ):
                    num_kv = attn_module.num_kv_heads
                    num_q = attn_module.num_heads
                    kv_dim = attn_module.qk_nope_head_dim + attn_module.v_head_dim
                    if loaded_weight.shape[0] == num_kv * kv_dim:
                        group_size = num_q // num_kv
                        loaded_weight = (
                            loaded_weight.view(num_kv, kv_dim, -1)
                            .repeat_interleave(group_size, dim=0)
                            .reshape(num_q * kv_dim, -1)
                        )

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
