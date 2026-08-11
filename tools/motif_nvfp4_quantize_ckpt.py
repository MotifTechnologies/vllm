# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize a Motif bf16 checkpoint's MoE experts to a serialized NVFP4 ckpt.

Produces the checkpoint consumed by the NVFP4 **direct-load** path: instead of
``--quantization modelopt_nvfp4`` re-quantizing bf16 weights at every server
start, the expert tensors are quantized once here and vLLM loads the packed
tensors as-is (auto-detected via the ``quantization_config`` this tool writes
into ``config.json`` — no serve-time flag needed).

Only the routed-expert weights (``moe.experts.gate_up_proj`` and
``moe.experts.down_proj``) are quantized, with the exact recipe the dynamic
loader uses: per-expert global scale ``448*6 / amax``, 1x16 E4M3 blockscales
along the input dim, E2M1 values packed two-per-byte. On a CUDA device
the default ``--backend kernel`` packs with the same
``ops.scaled_fp4_quant`` kernel as the dynamic loader (bit-identical
checkpoint); ``--backend torch`` is a CPU-capable fallback whose E2M1
rounding can differ from the kernel by one grid step on ~0.1% of values
(blockscales stay bit-identical). Everything else (attention, shared experts,
router/gate, norms, PolyNorm params, MTP) is copied verbatim in bf16.

Per expert tensor ``<name>`` the output carries:

  * ``<name>``                 packed E2M1  uint8          ``[E, X, K/2]``
  * ``<name>_weight_scale``    E4M3, linear layout          ``[E, X, K/16]``
  * ``<name>_weight_scale_2``  fp32 (``amax / (448*6)``)    ``[E]``

The blockscales are stored in linear (unswizzled) layout; the loader swizzles
them after expert placement. An ``nvfp4_act_scales.safetensors`` calibration
sidecar (see ``examples/offline_inference/motif_nvfp4_calibrate.py``) is
copied through if present and keeps working with the direct-load path.

Note: ``config.json``'s ``quantization_config`` is the authoritative marker
for the direct-load path — keep it when copying the checkpoint around. No
``hf_quant_config.json`` is written (a bare NVFP4 algo entry there would make
upstream vLLM claim the checkpoint for the incompatible ``modelopt_fp4``
path).

Usage:
    python tools/motif_nvfp4_quantize_ckpt.py \
        --input /models/motif3-bf16 --output /models/motif3-nvfp4 \
        [--device cuda] [--expert-chunk 32]
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.model_executor.layers.fused_moe.motif_nvfp4_experts import (
    quantize_experts_to_nvfp4_kernel,
    quantize_experts_to_nvfp4_ref,
)

_EXPERT_SUFFIXES = ("moe.experts.gate_up_proj", "moe.experts.down_proj")
_INDEX_NAME = "model.safetensors.index.json"


def _is_expert_weight(name: str) -> bool:
    return any(name.endswith(sfx) for sfx in _EXPERT_SUFFIXES)


def _model_shards(in_dir: str) -> list[str]:
    """Model weight shards: from the index if present, else all safetensors.

    Keeps non-model safetensors (e.g. the ``nvfp4_act_scales.safetensors``
    calibration sidecar) out of the quantization pass.
    """
    index_path = os.path.join(in_dir, _INDEX_NAME)
    if os.path.isfile(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        return sorted(set(index["weight_map"].values()))
    return sorted(
        f
        for f in os.listdir(in_dir)
        if f.endswith(".safetensors") and f != "nvfp4_act_scales.safetensors"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="bf16 Motif checkpoint dir")
    ap.add_argument("--output", required=True, help="output dir (created)")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device for the quantization math (default: cuda if available)",
    )
    ap.add_argument(
        "--backend",
        choices=("auto", "kernel", "torch"),
        default="auto",
        help="'kernel' = ops.scaled_fp4_quant (CUDA; bit-identical to the "
        "dynamic loader — the default on a cuda device); 'torch' = pure-torch "
        "reference (any device; E2M1 rounding at exact bucket boundaries can "
        "differ from the kernel by one grid step on ~0.1%% of values)",
    )
    ap.add_argument(
        "--expert-chunk", type=int, default=32,
        help="experts quantized at once (fp32 working-set memory vs speed; "
        "torch backend only)",
    )
    args = ap.parse_args()

    in_dir, out_dir = args.input, args.output
    device = torch.device(args.device)
    backend = args.backend
    if backend == "auto":
        backend = "kernel" if device.type == "cuda" else "torch"
    if backend == "kernel" and device.type != "cuda":
        raise SystemExit("--backend kernel requires --device cuda")
    print(f"[nvfp4] quantize backend: {backend} (device {device})")
    os.makedirs(out_dir, exist_ok=True)

    shards = _model_shards(in_dir)
    if not shards:
        raise SystemExit(f"No model safetensors shards found in {in_dir}")
    shard_set = set(shards)

    # Copy every non-shard file verbatim (tokenizer, *.py, jinja, calibration
    # sidecar, ...). config.json and the index are rewritten below.
    for f in os.listdir(in_dir):
        src = os.path.join(in_dir, f)
        if f in shard_set or f in ("config.json", _INDEX_NAME):
            continue
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, f))
    print(f"[nvfp4] copied non-shard files; {len(shards)} shards to process")

    new_weight_map: dict[str, str] = {}
    total_size = 0
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
                    src = t.to(device)
                    if backend == "kernel":
                        packed, scale, scale_2 = (
                            quantize_experts_to_nvfp4_kernel(src)
                        )
                    else:
                        packed, scale, scale_2 = quantize_experts_to_nvfp4_ref(
                            src, chunk=args.expert_chunk
                        )
                    tensors[name] = packed.cpu()
                    tensors[f"{name}_weight_scale"] = scale.cpu()
                    tensors[f"{name}_weight_scale_2"] = scale_2.cpu()
                    n_converted += 1
                else:
                    tensors[name] = t
        save_file(tensors, os.path.join(out_dir, shard), metadata=metadata)
        for name, t in tensors.items():
            new_weight_map[name] = shard
            total_size += t.numel() * t.element_size()
        print(f"[nvfp4] shard {i}/{len(shards)} {shard} done", flush=True)

    if n_converted == 0:
        raise SystemExit(
            "No routed-expert weights found — is this a Motif checkpoint?"
        )

    with open(os.path.join(out_dir, _INDEX_NAME), "w") as fh:
        json.dump(
            {
                "metadata": {"format": "pt", "total_size": total_size},
                "weight_map": new_weight_map,
            },
            fh,
            indent=2,
        )

    with open(os.path.join(in_dir, "config.json")) as fh:
        config = json.load(fh)
    # The direct-load marker: auto-selects --quantization modelopt_nvfp4 and
    # tells weight_utils.get_quant_config to build the direct-load config.
    # Deliberately no "quant_algo" key and no "producer" key — vLLM's config
    # normalizer maps producer.name == "modelopt" + an FP4 algo to upstream's
    # serialized modelopt_fp4 path (which does not know motif's expert
    # layout), and expects "producer" to be a dict when present.
    config["quantization_config"] = {
        "quant_method": "modelopt_nvfp4",
        "group_size": 16,
        "produced_by": "tools/motif_nvfp4_quantize_ckpt.py",
    }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    print(
        f"[nvfp4] DONE — quantized {n_converted} expert tensor(s) to NVFP4 "
        f"({total_size / 1e9:.1f} GB total). Output: {out_dir}\n"
        f"[nvfp4] Serve directly (no flag needed): vllm serve {out_dir}"
    )


if __name__ == "__main__":
    main()
