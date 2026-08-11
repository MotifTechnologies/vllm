# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MotifConfig for vLLM - Motif model with GDLA (Grouped Differential Latent Attention)."""

from transformers import PretrainedConfig


class MotifConfig(PretrainedConfig):
    model_type = "Motif"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling=None,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        sliding_window_pattern: str = "interleave",
        sliding_window_period: int = 2,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        # Differential Attention
        head_dim: int | None = None,
        num_noise_heads: int = 0,
        k_ratio: int = 1,
        # MoE
        num_experts: int = 0,
        experts_top_k: int = 2,
        num_shared_experts: int = 0,
        interleave_moe_layer_step: int = 0,
        moe_intermediate_size: int | None = None,
        score_func: str = "softmax",
        route_norm: bool = False,
        route_scale: float = 1.0,
        load_balance_coeff: float | None = None,
        score_before_experts: bool = False,
        _debug_force_load_balance: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.0,
        n_dense_first_layers: int = 0,
        # PolyNorm output scale (applied after activation, before w2/down_proj)
        polynorm_output_scale: float = 1.0,
        polynorm_output_scale_per_layer: dict | None = None,
        # Hard clamp [-c, c] applied to the routed-expert PolyNorm bias before
        # the activation (mirrors training GroupedExpertsPolyNorm.bias_clamp).
        # None = no clamp. Only the routed experts clamp the bias; the dense /
        # shared-expert MLPs (FeedForward) leave it unclamped, matching training.
        polynorm_bias_clamp: float | None = None,
        # MHC
        mhc_enabled: bool = False,
        mhc_expansion_rate: int = 4,
        mhc_identity_init: bool = False,
        mhc_sinkhorn_iters: int = 20,
        # DiffAttn / attention class
        diff_v2: bool = False,
        attention_cls: str = "basic",
        # GDLA
        q_lora_rank: int = 0,
        kv_lora_rank: int = 0,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        original_seq_len: int = 32768,
        rope_factor: float = 1.0,
        mscale: float = 1.0,
        # Attention output gating
        headwise_attn_output_gate: bool = False,
        elementwise_attn_output_gate: bool = False,
        # MTP (Multi-Token Prediction) speculative decoding
        num_nextn_predict_layers: int = 0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.sliding_window_pattern = sliding_window_pattern
        self.sliding_window_period = sliding_window_period
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout

        self.head_dim = head_dim
        self.num_noise_heads = num_noise_heads
        self.k_ratio = k_ratio

        self.num_experts = num_experts
        self.experts_top_k = experts_top_k
        self.num_shared_experts = num_shared_experts
        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.load_balance_coeff = load_balance_coeff
        self.score_before_experts = score_before_experts
        self._debug_force_load_balance = _debug_force_load_balance
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.n_dense_first_layers = n_dense_first_layers
        self.polynorm_output_scale = polynorm_output_scale
        self.polynorm_output_scale_per_layer = polynorm_output_scale_per_layer or {}
        self.polynorm_bias_clamp = polynorm_bias_clamp

        self.mhc_enabled = mhc_enabled
        self.mhc_expansion_rate = mhc_expansion_rate
        self.mhc_identity_init = mhc_identity_init
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters

        self.diff_v2 = diff_v2
        self.attention_cls = attention_cls
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        # qk_rope_head_dim / qk_nope_head_dim / v_head_dim: mirror the defaults
        # used inside the model (motif.py:MotifGDLAttention.__init__) and expose
        # qk_nope_head_dim as an attribute so vLLM core's get_mla_dims() can read
        # it directly from hf_text_config. Guarded for head_dim=None (the vLLM-
        # side default leaves head_dim unset until a checkpoint provides it).
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        if head_dim is not None:
            if self.qk_rope_head_dim is None:
                self.qk_rope_head_dim = head_dim // 2
            self.qk_nope_head_dim = head_dim - self.qk_rope_head_dim
            if self.v_head_dim is None:
                self.v_head_dim = head_dim
        self.original_seq_len = original_seq_len
        self.rope_factor = rope_factor
        self.mscale = mscale

        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        self.num_nextn_predict_layers = num_nextn_predict_layers

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
