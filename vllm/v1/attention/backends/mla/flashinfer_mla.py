# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType

logger = init_logger(__name__)

FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM


class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # cute-dsl MLA supports page_size 128 (block_num % (128/block_size)==0 always
        # holds for 128); the [32,64] cap is the trtllm-gen XQA constraint. motif keeps
        # block_size 128 so the hybrid full-attn/SWA KV grouping is unchanged.
        return [32, 64, 128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # FlashInfer MLA kernel requires qk_nope_head_dim in [64, 128, 192]
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim not in [64, 128, 192]:
                return (
                    "FlashInfer MLA kernel requires qk_nope_head_dim "
                    f"in [64, 128, 192], but got {qk_nope_head_dim}"
                )
        return None

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        # MOTIF FIX: "HND" is set GLOBALLY via set_kv_cache_layout() (selector.py)
        # and is read by EVERY backend, including motif's hybrid SWA layers
        # (FlashAttentionDiffKV). Forcing "HND" transposes the SWA KV cache stride
        # (NHD (0,1,2,3) -> HND (0,2,1,3)), corrupting the 39 SWA layers -> garbage
        # output, even though the MLA cache itself is head-agnostic (num_kv_heads=1,
        # so HND==NHD physically) and the cute-dsl MLA kernel reads it correctly
        # regardless (verified: forward_mqa torch-ref rel~0.003). The upstream "HND"
        # requirement exists for the trtllm-gen XQA kernel; the cute-dsl backend does
        # not need it. Return None (like CutlassMLABackend) so the global layout
        # stays the default NHD and the SWA layers are laid out correctly.
        import os as _os

        if _os.environ.get("MOTIF_FLASHINFER_MLA_BACKEND", "cute-dsl") == "cute-dsl":
            return None
        return "HND"


g_fi_workspace = torch.zeros(
    FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
    dtype=torch.uint8,
    device="cuda",
)


class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLAImpl"
            )

        self._workspace_buffer = g_fi_workspace
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        # trtllm API requires extra dimension q_len_per_request for MTP
        if attn_metadata.num_decode_tokens % attn_metadata.num_decodes != 0:
            logger.warning_once(
                """FlashInferMLAImpl got a query of uneven length.
                This usually indicates an issue in batch reordering
                or incorrect setup in dummy_run."""
            )
            q = q.unsqueeze(1)
        else:
            q = q.view(attn_metadata.num_decodes, -1, q.shape[-2], q.shape[-1])

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float

        # MOTIF: route MLA decode through flashinfer's cute-dsl backend.
        # The default trtllm-gen (XQA) kernel rejects motif's num_heads=80
        # ("numHeadsQ/numHeadsKv is not supported"); cute-dsl supports
        # num_heads<128 via folding (flashinfer PR#3309) AND keeps bf16 KV
        # (byte-identical on B200, PR#3664). num_spec=1 => verify q_len=2 (<=4),
        # so the modular-vs-monolithic eligibility gate (#3664, q_len>4) is not hit.
        # Env override lets us A/B "auto"/"trtllm-gen" during bring-up.
        import os as _os

        _motif_fi_backend = _os.environ.get("MOTIF_FLASHINFER_MLA_BACKEND", "cute-dsl")
        o = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            # cute-dsl MLA kernel requires an int8 workspace; trtllm-gen used uint8.
            # Same 1-byte element, so a view is free and lossless.
            workspace_buffer=(
                self._workspace_buffer.view(torch.int8)
                if _motif_fi_backend == "cute-dsl"
                else self._workspace_buffer
            ),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=attn_metadata.decode.block_table,
            seq_lens=attn_metadata.decode.seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            backend=_motif_fi_backend,
        )

        # Flatten the output for consistent shape
        o = o.view(-1, o.shape[-2], o.shape[-1])

        # TODO: Return LSE pending support from Flashinfer API:
        # https://github.com/flashinfer-ai/flashinfer/pull/1566
        return o, None
