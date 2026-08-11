# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Motif MTP (Multi-Token Prediction) model for speculative decoding.

Mirrors the training code in llm_training/motif/model/model.py (add_mtp_layer type).

Training forward for each MTP step k (1-indexed):
    next_h    = mtp_embed_norm[k](embed(t_{i+k}))   # enorm — normalise next-token embed
    concat_h  = cat([h, next_h], dim=-1)
    h         = mtp_layer[k](concat_h)               # eh_proj  Linear(2D -> D)
    h, _      = transformer_block[k](h, ...)
    h         = mtp_hidden_norm[k](h)                # final_layernorm
    logits    = model.output(h)                      # shared lm_head

vs Step3p5:
    enorm + hnorm applied to both embed and hidden before eh_proj.
    Per-layer shared_head (norm + lm_head) instead of one shared lm_head.

At inference, embed(t_{i+k}) comes from eagle.py which pre-shifts input_ids.

HF checkpoint weight names (0-indexed mtp_layers):
    model.mtp_layers.{i}.embed_norm.weight       -> enorm
    model.mtp_layers.{i}.input_proj.weight       -> eh_proj
    model.mtp_layers.{i}.final_layernorm.weight
    model.mtp_layers.{i}.self_attn.*             -> mtp_block.self_attn.*
    model.mtp_layers.{i}.mlp.*                   -> mtp_block.mlp.*
    model.mtp_layers.{i}.input_layernorm.weight  -> mtp_block.input_layernorm.weight
    model.mtp_layers.{i}.post_attention_layernorm.weight -> mtp_block.*

vLLM internal names shift 0-indexed HF MTP index by num_hidden_layers so KV cache
slot assignment is contiguous with base model layers.
"""

import logging
import os
import re
import types
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.motif import MotifConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec

from .motif import MotifDecoderLayer, MotifMoEFused
from .utils import maybe_prefix

logger = logging.getLogger(__name__)
_MTP_DEBUG = bool(os.environ.get("MOTIF_MTP_DEBUG"))


def _make_mtp_hf_config(config: MotifConfig) -> MotifConfig:
    """Return a MotifConfig clone for MTP transformer blocks.

    Training uses:
        dense_args = replace(model_args,
            interleave_moe_layer_step=0,
            sliding_window_pattern="all",   # ALL MTP layers are SWA
            mhc_enabled=False)

    MTP layers are SWA: swa_rope_theta (plain RoPE, no YaRN), no mscale.
    """
    import copy
    mtp_cfg = copy.copy(config)
    mtp_cfg.interleave_moe_layer_step = 0
    mtp_cfg.mhc_enabled = False
    mtp_cfg.use_sliding_window = True
    mtp_cfg.sliding_window_pattern = "all"
    return mtp_cfg


def _patch_mtp_attn_kv_cache_spec(block: nn.Module) -> None:
    """No-op on the v0.20.2 stack — kept for interface stability / depth>1.

    History: an older base (39 base SWA + 3 MTP SWA layers) forced MTP attention to
    report FullAttentionSpec so the 3 MTP layers wouldn't split across SWA subgroups.

    On the v0.20.2 stack that override is WRONG and breaks KV-cache reshape: Motif
    uses an asymmetric DiffKV backend (qk=192, v=128), so the base full-attn group's
    per-layer KV page uses head_size 384 while a fabricated FullAttentionSpec built
    from self_attn.head_size reports 320 → `raw_tensor.view([..., 320])` fails on a
    384-sized buffer. With depth-1 there is a single MTP layer, so the original
    subgroup-splitting problem cannot occur; the MTP layer keeps its natural
    SlidingWindowSpec and joins the base SWA layers' group (matching head_size).

    For depth>1, revisit: preserve the layer's natural spec dims (block_size /
    num_kv_heads / head_size / dtype) and only adjust grouping if validation fails.
    """
    return


class MotifMultiTokenPredictorLayer(nn.Module):
    """One MTP predictor step.

    vs Step3p5:
        - enorm only (no hnorm on previous_hidden_states)
        - shared lm_head at MotifMTP level (no per-layer shared_head)
    """

    def __init__(
        self,
        config: MotifConfig,
        layer_idx: int,
        vllm_config: VllmConfig,
        mtp_hf_config: MotifConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.mtp_block = MotifDecoderLayer(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mtp_block",
            layer_idx=layer_idx,
            config_override=mtp_hf_config,
        )
        _patch_mtp_attn_kv_cache_spec(self.mtp_block)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        next_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            positions:               token positions [num_tokens]
            previous_hidden_states:  h[i] from main model or previous MTP step
            next_hidden_states:      embed(t_{i+1}), pre-shifted by eagle.py
        """
        if _MTP_DEBUG and not torch.compiler.is_compiling():
            logger.warning(
                "[MTP DEBUG] prev_h: shape=%s mean=%.4f std=%.4f  "
                "next_h: mean=%.4f std=%.4f  pos=%s",
                previous_hidden_states.shape,
                previous_hidden_states.float().mean().item(),
                previous_hidden_states.float().std().item(),
                next_hidden_states.float().mean().item(),
                next_hidden_states.float().std().item(),
                positions[:4].tolist(),
            )

        next_hidden_states = self.enorm(next_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([previous_hidden_states, next_hidden_states], dim=-1)
        )
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        hidden_states = residual + hidden_states
        return self.final_layernorm(hidden_states)


class MotifMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: MotifConfig = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        # Kept for embed_input_ids() interface compatibility with EAGLE runner
        # and to load embed_tokens weights from checkpoint without errors.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        mtp_hf_config = _make_mtp_hf_config(config)

        self.layers = nn.ModuleDict(
            {
                str(idx): MotifMultiTokenPredictorLayer(
                    config=config,
                    layer_idx=idx,
                    vllm_config=vllm_config,
                    mtp_hf_config=mtp_hf_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        last_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:              shifted token ids — input_ids[i] = t_{i+1}
                                    (eagle.py already shifts them)
            previous_hidden_states: h from main model (step 0) or prev MTP step
            inputs_embeds:          unused, kept for interface compat
            spec_step_idx:          which MTP head to run (0-indexed)
        """
        # next_hidden_states = embed(t_{i+1}); matches training at mtp_embed_alpha=1
        if input_ids is not None:
            next_hidden_states = self.embed_tokens(input_ids)
        else:
            next_hidden_states = previous_hidden_states.new_zeros(
                previous_hidden_states.shape
            )

        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            positions,
            previous_hidden_states,
            next_hidden_states,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Motif uses one shared lm_head (vs Step3p5's per-layer shared_head)
        return self.logits_processor(lm_head, hidden_states)


@support_torch_compile
class MotifMTP(nn.Module):
    """Top-level Motif MTP draft model for vLLM speculative decoding."""

    # Detected by eagle.py to pass spec_step_idx in the sequential draft loop.
    _mtp_uses_spec_step_idx: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: MotifConfig = vllm_config.model_config.hf_config
        self.model = MotifMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Shared lm_head across all MTP steps (vs Step3p5's per-layer shared_head)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        last_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _MTP_DEBUG and not torch.compiler.is_compiling():
            logger.warning(
                "[MTP DEBUG] MotifMTP.forward spec_step_idx=%d h_is_none=%s",
                spec_step_idx, hidden_states is None,
            )
        if hidden_states is None:
            # Profiling dummy pass
            num_tokens = (
                input_ids.shape[0] if input_ids is not None
                else inputs_embeds.shape[0]  # type: ignore[union-attr]
            )
            param = next(self.parameters())
            hidden_states = torch.zeros(
                num_tokens, self.config.hidden_size,
                dtype=param.dtype, device=param.device,
            )
        return self.model(
            input_ids, positions, hidden_states,
            inputs_embeds, spec_step_idx, last_token_indices,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        logits = self.model.compute_logits(hidden_states, self.lm_head, spec_step_idx)
        if _MTP_DEBUG and not torch.compiler.is_compiling():
            logger.warning(
                "[MTP DEBUG] compute_logits: h mean=%.4f std=%.4f  "
                "logits mean=%.4f std=%.4f argmax=%s",
                hidden_states.float().mean().item(),
                hidden_states.float().std().item(),
                logits.float().mean().item(),
                logits.float().std().item(),
                logits.float().argmax(dim=-1).tolist(),
            )
        return logits

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _map_hf_name(self, name: str) -> str | None:
        """Map an HF checkpoint weight name to the vLLM parameter path.

        HF format (0-indexed):
          model.mtp_layers.{i}.embed_norm.weight
          model.mtp_layers.{i}.input_proj.weight
          model.mtp_layers.{i}.final_layernorm.weight
          model.mtp_layers.{i}.self_attn.*
          model.mtp_layers.{i}.mlp.*
          model.mtp_layers.{i}.input_layernorm.weight
          model.mtp_layers.{i}.post_attention_layernorm.weight

        vLLM format (shifted by num_hidden_layers):
          model.layers.{i+N}.enorm.weight
          model.layers.{i+N}.eh_proj.weight
          model.layers.{i+N}.final_layernorm.weight
          model.layers.{i+N}.mtp_block.self_attn.*
          model.layers.{i+N}.mtp_block.mlp.*
          model.layers.{i+N}.mtp_block.input_layernorm.weight
          model.layers.{i+N}.mtp_block.post_attention_layernorm.weight
        """
        if name.startswith("model.mtp_layers."):
            m = re.match(r"model\.mtp_layers\.(\d+)\.(.*)", name)
            if not m:
                return name
            hf_idx = int(m.group(1))
            rest = m.group(2)
            vllm_idx = hf_idx + self.config.num_hidden_layers

            # Rename HF param names to vLLM internal names
            _rename = {
                "embed_norm": "enorm",
                "input_proj": "eh_proj",
            }
            for hf_frag, vllm_frag in _rename.items():
                if rest.startswith(hf_frag):
                    rest = vllm_frag + rest[len(hf_frag):]
                    break

            # Direct per-layer params: no mtp_block prefix
            _direct = ("enorm", "eh_proj", "final_layernorm")
            if any(rest.startswith(p) for p in _direct):
                return f"model.layers.{vllm_idx}.{rest}"

            # Transformer block params: add mtp_block prefix
            return f"model.layers.{vllm_idx}.mtp_block.{rest}"

        if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
            return name

        return None  # skip non-MTP weights (belong to the base model)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Attention weight renames: HF Titan names -> vLLM internal names
        attn_rename = {
            "self_attn.wq_a":        "self_attn.q_a_proj",
            "self_attn.q_norm":      "self_attn.q_a_layernorm",
            "self_attn.wq_b.":       "self_attn.q_b_proj.",
            "self_attn.wq_b_gate":   "self_attn.q_b_gate",
            "self_attn.wkv_a":       "self_attn.kv_a_proj_with_mqa",
            "self_attn.kv_norm":     "self_attn.kv_a_layernorm",
            "self_attn.wkv_b":       "self_attn.kv_b_proj",
            "self_attn.lambda_proj": "self_attn.lambda_proj",
            "self_attn.wo":          "self_attn.o_proj",
            "moe.router.gate":       "moe.gate",
            "moe.expert_bias":       "moe.e_score_correction_bias",
        }

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            name = self._map_hf_name(name)
            if name is None:
                continue

            for ckpt_frag, vllm_frag in attn_rename.items():
                if ckpt_frag in name:
                    name = name.replace(ckpt_frag, vllm_frag)
                    break

            # Handle MoE expert weights inside MTP blocks (dense by default,
            # but guard here in case a checkpoint has them).
            moe_module = self._find_moe_for_param(name)
            if isinstance(moe_module, MotifMoEFused):
                if name.endswith(".moe.e_score_correction_bias"):
                    name = name.replace(
                        ".moe.e_score_correction_bias",
                        ".moe.gate.e_score_correction_bias",
                    )
                if self._load_fused_moe_expert_weight(
                    name, loaded_weight, moe_module, params_dict, loaded_params
                ):
                    continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # fp8 attention scale params (q_scale/k_scale/v_scale/prob_scale) are created by
        # the quantization method with default (identity) values; a bf16 checkpoint has
        # none — identical to the base model, which serves fine. Mark them loaded so
        # vLLM's strict uninitialized-weights check (default_loader) doesn't fail the
        # draft model.
        _FP8_ATTN_SCALES = ("q_scale", "k_scale", "v_scale", "prob_scale")
        for pname in params_dict:
            if pname not in loaded_params and pname.rsplit(".", 1)[-1] in _FP8_ATTN_SCALES:
                loaded_params.add(pname)

        logger.warning(
            "[MTP] load_weights: loaded %d / %d params",
            len(loaded_params), len(params_dict),
        )
        unloaded = [k for k in params_dict if k not in loaded_params]
        if unloaded:
            logger.warning("[MTP] unloaded params (first 10): %s", unloaded[:10])
        return loaded_params

    # ------------------------------------------------------------------
    # MoE helpers (mirrored from MotifForCausalLM)
    # ------------------------------------------------------------------

    def _find_moe_for_param(self, name: str):
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
        moe_module: MotifMoEFused,
        params_dict: dict,
        loaded_params: set,
    ) -> bool:
        fused_layer = moe_module.experts

        if "moe.experts.gate_up_proj" in name:
            target = name.replace("moe.experts.gate_up_proj", "moe.experts.w13_weight")
            param = params_dict.get(target)
            if param is None:
                return True
            E_ckpt, two_I, _ = loaded_weight.shape
            I = two_I // 2
            for ge in range(E_ckpt):
                param.weight_loader(param, loaded_weight[ge, :I, :], target, "w1", ge)
                param.weight_loader(param, loaded_weight[ge, I:, :], target, "w3", ge)
            loaded_params.add(target)
            return True

        if "moe.experts.down_proj" in name:
            target = name.replace("moe.experts.down_proj", "moe.experts.w2_weight")
            param = params_dict.get(target)
            if param is None:
                return True
            for ge in range(loaded_weight.shape[0]):
                param.weight_loader(param, loaded_weight[ge], target, "w2", ge)
            loaded_params.add(target)
            return True

        for ckpt_frag, vllm_frag in (
            ("moe.experts.act_fn.weight", "moe.experts.act_fn_weight"),
            ("moe.experts.act_fn.bias",   "moe.experts.act_fn_bias"),
        ):
            if ckpt_frag in name:
                target = name.replace(ckpt_frag, vllm_frag)
                param = params_dict.get(target)
                if param is None:
                    return True
                expert_map = getattr(fused_layer, "_expert_map", None)
                if expert_map is None:
                    param.data.copy_(loaded_weight.to(param.dtype))
                else:
                    for ge in range(loaded_weight.shape[0]):
                        le = int(expert_map[ge].item())
                        if le < 0:
                            continue
                        param.data[le].copy_(loaded_weight[ge].to(param.dtype))
                loaded_params.add(target)
                return True

        return False
