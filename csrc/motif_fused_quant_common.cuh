// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared device helpers for the motif3 fused MoE quant kernels
// (motif_fused_poly_quant_kernel.cu, motif_fused_rmsnorm_quant_kernel.cu):
// a one-CTA-per-row block reduction (shared-memory tree, broadcasts the result
// to all lanes via sm[0]) and the 1x128 e4m3 scale computation.

#pragma once
#include <cuda_runtime.h>

namespace vllm {

// Block reduction over BLOCK threads; `sm` is a [BLOCK] shared buffer. Result
// is returned on every thread (read from sm[0]).
template <int BLOCK>
__device__ __forceinline__ float motif_block_reduce_sum(float v, float* sm,
                                                        int tid) {
  sm[tid] = v;
  __syncthreads();
#pragma unroll
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) sm[tid] += sm[tid + s];
    __syncthreads();
  }
  float r = sm[0];
  __syncthreads();
  return r;
}

template <int BLOCK>
__device__ __forceinline__ float motif_block_reduce_max(float v, float* sm,
                                                        int tid) {
  sm[tid] = v;
  __syncthreads();
#pragma unroll
  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);
    __syncthreads();
  }
  float r = sm[0];
  __syncthreads();
  return r;
}

// Per-group e4m3 scale: amax / fp8_max, rounded up to a power of two (UE8M0)
// when use_ue8m0 (the DeepGEMM E8M0 recipe).
__device__ __forceinline__ float motif_fp8_scale(float amax, float fp8_max,
                                                 float quant_eps,
                                                 bool use_ue8m0) {
  amax = fmaxf(amax, quant_eps);
  float scale = amax / fp8_max;
  return use_ue8m0 ? exp2f(ceilf(log2f(scale))) : scale;
}

}  // namespace vllm
