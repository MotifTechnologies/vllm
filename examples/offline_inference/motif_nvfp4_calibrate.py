# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate NVFP4 activation scales for a Motif MoE checkpoint.

``--quantization modelopt_nvfp4`` quantizes the routed experts to NVFP4 at load.
Uncalibrated, the per-expert *activation* global scale defaults to 1.0, which
pushes small-magnitude activation blocks into E4M3 subnormals and costs accuracy
on long-reasoning benchmarks. This script runs representative prompts through the
engine with amax accumulation enabled, then writes a per-global-expert sidecar
(``nvfp4_act_scales.safetensors``) that the loader picks up on the next launch.

Usage (SM100 / Blackwell required):

    VLLM_MOTIF_NVFP4_CALIBRATE=1 python \
        examples/offline_inference/motif_nvfp4_calibrate.py \
        --model /path/to/motif-bf16-checkpoint \
        --prompts-file calib_prompts.txt \
        --output /path/to/motif-bf16-checkpoint/nvfp4_act_scales.safetensors \
        --tensor-parallel-size 1 --enable-expert-parallel

Then serve/RL with the SAME checkpoint dir (the sidecar is auto-discovered), or
point ``VLLM_MOTIF_NVFP4_ACT_SCALES`` at the output file. ``VLLM_MOTIF_NVFP4_
CALIBRATE`` must be UNSET for normal serving.

Notes:
- Calibration must run in eager mode (``enforce_eager=True``, set below) so the
  amax buffers are updated and readable; cudagraph capture is skipped.
- Use a calibration set that matches the target distribution — include the
  long-reasoning prompts where the accuracy drop shows up.
"""

import argparse
import json
import os

from vllm import LLM, SamplingParams


def _load_calib_items(path: str) -> list[tuple[str, object]]:
    """Load calibration records as ``("chat", messages)`` or ``("text", str)``.

    ``.jsonl``: each line is a record; SFT-style ``conversations`` become chat
    items, ``prompt``/``text`` become text items. Otherwise the file is treated
    as one raw prompt per line.
    """
    items: list[tuple[str, object]] = []
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict) and rec.get("conversations"):
                    items.append(("chat", rec["conversations"]))
                elif isinstance(rec, dict) and (rec.get("prompt") or rec.get("text")):
                    items.append(("text", rec.get("prompt") or rec.get("text")))
                elif isinstance(rec, str):
                    items.append(("text", rec))
    else:
        with open(path) as f:
            for line in f:
                if line.strip():
                    items.append(("text", line.rstrip("\n")))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="bf16 Motif checkpoint dir")
    parser.add_argument(
        "--prompts-file",
        required=True,
        help="calibration prompts. A .txt file (one raw prompt per line) OR a "
        ".jsonl file whose records carry SFT-style 'conversations' "
        "([{role, content}, ...]) — rendered through the model's chat template "
        "and processed as a single prefill so the experts see the full "
        "(incl. long-reasoning) token distribution. Records with 'prompt'/"
        "'text' are also accepted.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="cap the number of calibration records used (after any sampling).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="sidecar path (default: <model>/nvfp4_act_scales.safetensors)",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="cap the KV context (calibration prompts are short; keeps the "
        "huge MLA model's KV allocation small). Default: model config.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85
    )
    args = parser.parse_args()

    if os.environ.get("VLLM_MOTIF_NVFP4_CALIBRATE") != "1":
        raise SystemExit(
            "Set VLLM_MOTIF_NVFP4_CALIBRATE=1 before running so the experts "
            "register amax accumulators."
        )

    output = args.output or os.path.join(args.model, "nvfp4_act_scales.safetensors")
    items = _load_calib_items(args.prompts_file)
    if args.max_prompts is not None:
        items = items[: args.max_prompts]
    if not items:
        raise SystemExit(f"No prompts found in {args.prompts_file}")
    print(f"[calib] {len(items)} calibration records; scales -> {output}")

    llm_kwargs = dict(
        model=args.model,
        quantization="modelopt_nvfp4",
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,  # amax buffers need eager execution
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    # The amax dump ships a callable to the workers via collective_rpc; the
    # default msgspec serializer refuses functions, so opt into the pickle
    # fallback (offline calibration run — not a serving surface).
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(**llm_kwargs)

    # Render to token ids up front so full conversations (incl. the assistant
    # reasoning) are processed as ONE prefill — that is the token distribution
    # the experts see at inference. Truncate to the KV cap; calibration only
    # needs a representative pass, not the full sequence.
    tok = llm.get_tokenizer()
    max_len = (args.max_model_len or 8192) - args.max_tokens - 1
    token_prompts, skipped = [], 0
    for kind, payload in items:
        try:
            if kind == "chat":
                # return_dict=False: transformers v5 defaults to returning a
                # BatchEncoding here, whose len() is its KEY count (2) — the
                # truncation below and vLLM's prompt_token_ids need the plain
                # id list.
                ids = tok.apply_chat_template(
                    payload,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=False,
                )
            else:
                ids = tok(payload).input_ids
        except Exception:
            skipped += 1
            continue
        if len(ids) > max_len:
            ids = ids[:max_len]  # calibration truncation
        if ids:
            token_prompts.append({"prompt_token_ids": ids})
    if skipped:
        print(f"[calib] skipped {skipped} record(s) that failed to render")
    print(f"[calib] prefilling {len(token_prompts)} sequence(s) for amax")

    # max_tokens is tiny — prefill over the prompt is what accumulates amax.
    llm.generate(
        token_prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )

    # Reach the underlying model and dump on every worker (dump_nvfp4_act_scales
    # all-reduces across ranks and writes only on rank 0).
    def _dump(worker):
        from vllm.model_executor.layers.fused_moe.motif_nvfp4_experts import (
            dump_nvfp4_act_scales as _d,
        )

        return _d(worker.model_runner.model, output)

    # V1 engine: the model lives in worker processes — dump via the LLM-level
    # collective_rpc (llm_engine has no model_executor to reach directly).
    n = llm.collective_rpc(_dump)
    n_layers = n[0] if isinstance(n, list) else n

    print(f"[calib] done — wrote activation scales for {n_layers} MoE layer(s)")


if __name__ == "__main__":
    main()
