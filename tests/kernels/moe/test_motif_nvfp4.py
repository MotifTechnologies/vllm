# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the motif dynamic NVFP4 MoE path (--quantization modelopt_nvfp4).

The GPU tests require SM100+ (CUTLASS NVFP4); the config-plumbing test runs
anywhere.
"""

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import dequantize_nvfp4_to_dtype
from vllm.platforms import current_platform


def test_modelopt_nvfp4_dynamic_config_plumbing():
    """--quantization modelopt_nvfp4 resolves to the dynamic config class."""
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMxFp8Config,
        ModelOptNvFp4DynamicConfig,
    )

    quant_cls = get_quantization_config("modelopt_nvfp4")
    assert quant_cls is ModelOptNvFp4DynamicConfig

    # Dynamic sentinel (set by weight_utils.get_quant_config) builds the
    # non-serialized config that loads bf16 weights.
    cfg = quant_cls.from_config({"_nvfp4_dynamic": True})
    assert isinstance(cfg, ModelOptNvFp4DynamicConfig)
    # Shares the MXFP8 dynamic plumbing (bf16 load path in MotifMoEFused).
    assert isinstance(cfg, ModelOptMxFp8Config)
    assert not cfg.is_checkpoint_mxfp8_serialized
    assert cfg.get_name() == "modelopt_nvfp4"

    # Never auto-detected from a checkpoint's hf_quant_config.
    assert quant_cls.override_quantization_method({}, "modelopt_nvfp4") is None


def test_modelopt_nvfp4_direct_config_plumbing():
    """The _nvfp4_direct sentinel builds the direct-load config."""
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4DynamicConfig,
    )

    quant_cls = get_quantization_config("modelopt_nvfp4")

    direct = quant_cls.from_config({"_nvfp4_direct": True})
    assert isinstance(direct, ModelOptNvFp4DynamicConfig)
    assert direct.direct_load
    assert direct.get_name() == "modelopt_nvfp4"
    # Same "FusedMoE method owns the weights" plumbing as dynamic mode.
    assert not direct.is_checkpoint_mxfp8_serialized

    # The serialized checkpoint's config.json quantization_config (as written
    # by tools/motif_nvfp4_quantize_ckpt.py) — get_quant_config feeds it to
    # from_config verbatim, so the marker alone must select direct load.
    ckpt_form = quant_cls.from_config(
        {"quant_method": "modelopt_nvfp4", "group_size": 16}
    )
    assert ckpt_form.direct_load

    dynamic = quant_cls.from_config({"_nvfp4_dynamic": True})
    assert not dynamic.direct_load


_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Packed uint8 [..., K/2] -> fp32 values [..., K] (low nibble first)."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    mag = _E2M1_VALUES[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def _load_fakequant_tool():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "motif_nvfp4_fakequant_ckpt.py"
    )
    if not path.is_file():
        pytest.skip("tools/motif_nvfp4_fakequant_ckpt.py not found")
    spec = importlib.util.spec_from_file_location("motif_nvfp4_fakequant", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_quantize_experts_to_nvfp4_ref_cpu():
    """Pure-torch packer: shapes/dtypes, and dequant matches the independent
    QDQ reference from tools/motif_nvfp4_fakequant_ckpt.py (CPU, no GPU
    needed). The bit-level kernel comparison is the GPU test below."""
    from vllm.model_executor.layers.fused_moe.motif_nvfp4_experts import (
        _SF_FULL_RANGE,
        quantize_experts_to_nvfp4_ref,
    )

    torch.manual_seed(11)
    E, X, K = 4, 128, 256
    w = torch.randn(E, X, K, dtype=torch.bfloat16)
    w *= torch.logspace(-2, 1, E).view(E, 1, 1).to(w.dtype)

    packed, scale, scale_2 = quantize_experts_to_nvfp4_ref(w, chunk=3)

    assert packed.shape == (E, X, K // 2) and packed.dtype == torch.uint8
    assert scale.shape == (E, X, K // 16)
    assert scale.dtype == torch.float8_e4m3fn
    assert scale_2.shape == (E,) and scale_2.dtype == torch.float32

    amax = w.to(torch.float32).abs().amax(dim=(1, 2)).clamp(min=1e-12)
    torch.testing.assert_close(scale_2, amax / _SF_FULL_RANGE)

    # Dequantize: value * blockscale / gscale, gscale = 1 / scale_2.
    vals = _unpack_e2m1(packed).reshape(E, X, K // 16, 16)
    deq = scale.to(torch.float32) * scale_2.view(E, 1, 1)
    w_dq = (vals * deq.unsqueeze(-1)).reshape(E, X, K)

    fq = _load_fakequant_tool()
    # The fakequant tool is an independent QDQ implementation of the same
    # recipe; its dequant multiplier differs by <= 1 fp32 ulp (sf/gscale vs
    # sf * scale_2), so compare with a small tolerance. Layout bugs (nibble
    # order, scale placement, signs) would blow far past it. The bit-exact
    # comparison against the CUDA kernel is the SM100 test below.
    w_qdq = fq._nvfp4_qdq_experts(w.clone(), chunk=2).to(torch.float32)
    torch.testing.assert_close(
        w_dq.to(torch.bfloat16).to(torch.float32),
        w_qdq,
        rtol=1e-2,
        atol=1e-2,
    )


def test_gscale_from_amax_and_alpha_fold():
    """gscale maps amax -> E4M3 full range; the epilogue fold is invertible."""
    from vllm.model_executor.layers.fused_moe import motif_nvfp4_experts as M

    amax = torch.tensor([2.0, 4.0, 0.5])
    gscale = M._gscale_from_amax(amax)
    torch.testing.assert_close(gscale, M._SF_FULL_RANGE / amax)

    # Epilogue alpha = pure_weight_alpha / a_gscale must undo exactly:
    # (alpha_folded * a_gscale) == pure weight alpha.
    w_alpha_pure = torch.tensor([0.5, 0.25, 4.0])
    folded = w_alpha_pure / gscale
    torch.testing.assert_close(folded * gscale, w_alpha_pure)

    # a_gscale == 1 (uncalibrated) leaves the alpha as the pure weight term.
    ones = torch.ones_like(w_alpha_pure)
    torch.testing.assert_close(w_alpha_pure / ones, w_alpha_pure)


def test_act_scales_sidecar_load_and_slice(tmp_path, monkeypatch):
    """Per-GLOBAL sidecar is sliced to a rank's locals via expert_map."""
    from safetensors.torch import save_file

    from vllm.model_executor.layers.fused_moe import motif_nvfp4_experts as M

    e_global = 4
    a13 = torch.arange(1, e_global + 1, dtype=torch.float32)  # [1,2,3,4]
    a2 = a13 * 10
    path = str(tmp_path / "scales.safetensors")
    save_file({"L.a13_gscale": a13, "L.a2_gscale": a2}, path)

    monkeypatch.setenv("VLLM_MOTIF_NVFP4_ACT_SCALES", path)
    M._act_scales_cache.clear()
    cpu = torch.device("cpu")

    # Rank holds global experts 2, 3 in local slots 0, 1.
    expert_map = torch.tensor([-1, -1, 0, 1])
    g13, g2 = M._local_gscales_for_layer("L", 2, expert_map, cpu)
    torch.testing.assert_close(g13, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(g2, torch.tensor([30.0, 40.0]))

    # No expert_map (single rank): global == local.
    g13s, _ = M._local_gscales_for_layer("L", 4, None, cpu)
    torch.testing.assert_close(g13s, a13)

    # Layer absent from the sidecar -> None (loader falls back to ones()).
    assert M._local_gscales_for_layer("missing", 2, expert_map, cpu) is None
    M._act_scales_cache.clear()


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="NVFP4 requires compute capability 10.0+",
)
@pytest.mark.parametrize("E,X,K", [(8, 256, 512), (4, 512, 2048)])
@torch.inference_mode()
def test_quantize_experts_to_nvfp4(E: int, X: int, K: int):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.motif_nvfp4_experts import (
        FLOAT4_E2M1_MAX,
        FLOAT8_E4M3_MAX,
        _quantize_experts_to_nvfp4,
    )

    torch.manual_seed(7)
    device = torch.device("cuda")
    w = torch.randn(E, X, K, dtype=torch.bfloat16, device=device)
    # Give experts distinct ranges so per-expert global scales differ.
    w *= torch.logspace(-2, 1, E, device=device).view(E, 1, 1).to(w.dtype)

    w_fp4, w_sf, alphas = _quantize_experts_to_nvfp4(w)

    assert w_fp4.shape == (E, X, K // 2) and w_fp4.dtype == torch.uint8
    assert w_sf.shape == (E, X, K // 16)
    assert w_sf.dtype == torch.float8_e4m3fn
    assert alphas.shape == (E,) and alphas.dtype == torch.float32

    # Recompute the global scale exactly as the helper does (a
    # double-reciprocal through alphas is not bit-exact in fp32).
    amax = w.abs().amax(dim=(1, 2)).to(torch.float32).clamp(min=1e-12)
    global_scales = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    torch.testing.assert_close(alphas, 1.0 / global_scales)

    for e in range(E):
        # Layout check: quantize-unswizzled + swizzle_blockscale must be
        # bit-identical to the kernel's directly-swizzled output.
        q_ref, sf_ref = ops.scaled_fp4_quant(
            w[e], global_scales[e], is_sf_swizzled_layout=True
        )
        assert torch.equal(w_fp4[e], q_ref)
        assert torch.equal(
            w_sf[e].view(torch.uint8).flatten(),
            sf_ref.view(torch.uint8).flatten(),
        )

        # Quality check: dequantized weight tracks the original within
        # NVFP4 error (loose relative-Frobenius bound).
        w_dq = dequantize_nvfp4_to_dtype(
            w_fp4[e], w_sf[e], global_scales[e], torch.float32, device
        )
        ref = w[e].to(torch.float32)
        rel_err = (w_dq - ref).norm() / ref.norm()
        assert rel_err < 0.15, f"expert {e}: rel_err={rel_err.item():.4f}"


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="NVFP4 requires compute capability 10.0+",
)
@pytest.mark.parametrize("E,X,K", [(8, 256, 512), (4, 512, 2048)])
@torch.inference_mode()
def test_direct_load_ckpt_matches_dynamic_quantization(E: int, X: int, K: int):
    """The offline tool's tensors reproduce the dynamic path bit-for-bit.

    Direct load = quantize_experts_to_nvfp4_ref (what
    tools/motif_nvfp4_quantize_ckpt.py serializes) + swizzle_blockscale at
    load + alpha := weight_scale_2. Dynamic = _quantize_experts_to_nvfp4
    (ops.scaled_fp4_quant). Both must produce identical runtime tensors.
    """
    from vllm.model_executor.layers.fused_moe.motif_nvfp4_experts import (
        _quantize_experts_to_nvfp4,
        quantize_experts_to_nvfp4_kernel,
        quantize_experts_to_nvfp4_ref,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    torch.manual_seed(3)
    device = torch.device("cuda")
    w = torch.randn(E, X, K, dtype=torch.bfloat16, device=device)
    w *= torch.logspace(-2, 1, E, device=device).view(E, 1, 1).to(w.dtype)

    dyn_fp4, dyn_sf_swizzled, dyn_alphas = _quantize_experts_to_nvfp4(w)

    # Kernel backend (the offline tool's GPU path): bit-identical to the
    # dynamic loader — same packing kernel, swizzle deferred to load time.
    ker_fp4, ker_sf_linear, ker_scale_2 = quantize_experts_to_nvfp4_kernel(w)
    assert torch.equal(ker_fp4, dyn_fp4)
    assert torch.equal(
        swizzle_blockscale(ker_sf_linear).view(torch.uint8).flatten(),
        dyn_sf_swizzled.view(torch.uint8).flatten(),
    )
    torch.testing.assert_close(ker_scale_2, dyn_alphas, rtol=0, atol=0)

    # Torch-ref backend (CPU fallback): blockscales and scale_2 bit-identical;
    # E2M1 values may differ from the kernel by one grid step where the fp32
    # scaled value lands exactly on a rounding-bucket boundary (~0.1% of
    # elements on randn — both roundings are valid nearest-neighbors).
    ref_fp4, ref_sf_linear, ref_scale_2 = quantize_experts_to_nvfp4_ref(w)
    assert torch.equal(
        ref_sf_linear.view(torch.uint8), ker_sf_linear.view(torch.uint8)
    )
    torch.testing.assert_close(ref_scale_2, ker_scale_2, rtol=0, atol=0)
    lo_r, hi_r = ref_fp4 & 0xF, (ref_fp4 >> 4) & 0xF
    lo_k, hi_k = ker_fp4 & 0xF, (ker_fp4 >> 4) & 0xF
    mismatch = (
        (lo_r != lo_k).sum() + (hi_r != hi_k).sum()
    ).item() / (2 * ker_fp4.numel())
    assert mismatch < 5e-3, f"ref/kernel nibble mismatch rate {mismatch:.2e}"
