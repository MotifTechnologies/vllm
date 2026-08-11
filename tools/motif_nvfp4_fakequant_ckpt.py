# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fake-quantize a Motif bf16 checkpoint's MoE experts to NVFP4 and back to bf16.

Diagnostic tool for the w4a16 question: "how much of the modelopt_nvfp4 accuracy
drop is the WEIGHT quantization vs the activation quantization?"

It round-trips ONLY the routed-expert weights (``moe.experts.gate_up_proj`` and
``moe.experts.down_proj``) through the exact NVFP4 recipe the vLLM loader uses
(``_quantize_experts_to_nvfp4``): per-expert global scale ``448*6 / amax``, 1x16
E4M3 block scales, E2M1 value grid. The result is written back as **bf16**, so
serving the output checkpoint WITHOUT ``--quantization`` (plain bf16) reproduces
the accuracy of FP4 weights + bf16 activations — i.e. w4a16. Everything else
(attention, shared experts, router/gate, norms, PolyNorm params) is copied
verbatim.

3-way comparison:
  * bf16 original            -> baseline
  * this output (bf16 serve) -> weight-only FP4 loss (w4a16)
  * --quantization modelopt_nvfp4 on the original -> weight+activation (w4a4)

Pure torch + safetensors; no CUDA, no vLLM import needed. Runs on a CPU node.
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_FP4_MAX = 6.0  # E2M1 max
_E4M3_MAX = 448.0  # float8_e4m3fn max
_BLOCK = 16  # NVFP4 block size
_EXPERT_SUFFIXES = ("moe.experts.gate_up_proj", "moe.experts.down_proj")


def _cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 (fp4) value grid. Mirrors vLLM's ``cast_to_fp4``."""
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def _nvfp4_qdq_experts(w: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """NVFP4 quantize-dequantize a stacked expert weight ``[E, X, K]`` (bf16).

    Per-expert global scale ``448*6 / amax`` (matches
    ``_quantize_experts_to_nvfp4``); 1x16 E4M3 block scales along K; E2M1 values.
    Chunked over experts to bound the fp32 working set. Returns bf16.
    """
    assert w.dim() == 3, f"expected [E, X, K], got {tuple(w.shape)}"
    E, X, K = w.shape
    assert K % _BLOCK == 0, f"K={K} not divisible by {_BLOCK}"
    out = torch.empty_like(w)
    for s in range(0, E, chunk):
        e = slice(s, min(s + chunk, E))
        x = w[e].to(torch.float32)  # [n, X, K]
        n = x.shape[0]
        amax = x.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)  # [n,1,1]
        gscale = (_E4M3_MAX * _FP4_MAX) / amax  # [n,1,1]

        xb = x.reshape(n, X, K // _BLOCK, _BLOCK)
        vmax = xb.abs().amax(dim=-1, keepdim=True)  # [n,X,K/blk,1]
        scale = (gscale.unsqueeze(-1) * (vmax / _FP4_MAX)).clamp(
            -_E4M3_MAX, _E4M3_MAX
        )
        # E4M3 rounding of the block scale (the real kernel stores it in e4m3).
        scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
        deq = scale / gscale.unsqueeze(-1)  # per-block dequant multiplier

        q = torch.clamp(xb / deq.clamp_min(1e-20), -_FP4_MAX, _FP4_MAX)
        q = _cast_to_fp4(q.contiguous())
        out[e] = (q * deq).reshape(n, X, K).to(w.dtype)
    return out


def _is_expert_weight(name: str) -> bool:
    return any(name.endswith(sfx) for sfx in _EXPERT_SUFFIXES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="bf16 Motif checkpoint dir")
    ap.add_argument("--output", required=True, help="output dir (created)")
    ap.add_argument(
        "--expert-chunk", type=int, default=32,
        help="experts processed at once (memory vs speed)",
    )
    args = ap.parse_args()

    in_dir, out_dir = args.input, args.output
    os.makedirs(out_dir, exist_ok=True)

    shards = sorted(f for f in os.listdir(in_dir) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"No .safetensors shards in {in_dir}")

    # Copy every non-shard file verbatim (config, tokenizer, index, *.py, jinja).
    for f in os.listdir(in_dir):
        src = os.path.join(in_dir, f)
        if f.endswith(".safetensors") or not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(out_dir, f))
    print(f"[fakequant] copied non-shard files; {len(shards)} shards to process")

    n_converted = 0
    for i, shard in enumerate(shards, 1):
        tensors: dict[str, torch.Tensor] = {}
        metadata = {"format": "pt"}
        with safe_open(os.path.join(in_dir, shard), framework="pt") as f:
            meta = f.metadata()
            if meta:
                metadata.update(meta)
            for name in f.keys():  # noqa: SIM118
                t = f.get_tensor(name)
                if _is_expert_weight(name):
                    t = _nvfp4_qdq_experts(t, chunk=args.expert_chunk)
                    n_converted += 1
                tensors[name] = t
        save_file(tensors, os.path.join(out_dir, shard), metadata=metadata)
        print(f"[fakequant] shard {i}/{len(shards)} {shard} done", flush=True)

    print(
        f"[fakequant] DONE — round-tripped {n_converted} expert tensor(s) to "
        f"NVFP4 and back to bf16. Output: {out_dir}\n"
        f"[fakequant] Serve WITHOUT --quantization (plain bf16) to measure w4a16."
    )

    # Sanity: index.json (if present) already points at the same shard names, so
    # it was copied as-is and stays valid.
    idx = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as fh:
            json.load(fh)  # parse-check only


if __name__ == "__main__":
    main()
