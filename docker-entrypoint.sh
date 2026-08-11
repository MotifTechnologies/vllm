#!/usr/bin/env bash
set -e

export NVSHMEM_DIR=/usr/local/nvshmem
export LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH}
export PATH=${NVSHMEM_DIR}/bin:${PATH}

# cache dirs live on the lustrefs volume mounted at runtime
mkdir -p /lustrefs/team-service/cache/{hf,vllm,triton,inductor,tilelang}

nvidia-smi -L
python -c "import deep_ep; print('[run] deep_ep OK')"

# Preflight (mirrors the serve YAML): fail fast if the baked wheel is not the
# motif3 one or the packed_scale op schema (quickopt csrc) is missing.
python - <<'EOF'
import vllm, torch, vllm._C  # noqa: F401
v = vllm.__version__
assert "motif3" in v, f"not a motif3 wheel: {v}"
s = str(torch.ops._C.grouped_poly_norm_fp8_quant.default._schema)
assert "packed_scale" in s, s
import vllm.model_executor.layers.mhc  # noqa: F401  (MHC ops registered)
print(f"[run] vllm {v} OK (packed_scale + mhc ops)")
EOF

exec "$@"
