/*
 * Grouped Fused-Mul Poly Norm activation kernel for the motif3 MoE.
 *
 * This kernel targets vLLM's Standard FusedMoE activation format (the format
 * produced by `TritonExperts` between GEMM1 and GEMM2):
 *
 *   intermediate_cache1 : [num_tokens, top_k, 2*I]   (gate || up after GEMM1)
 *   intermediate_cache2 : [num_tokens * top_k, I]    (activation output)
 *
 * Each row r in [0, num_tokens * top_k) corresponds to (token_idx, k_idx)
 * with row = token_idx * top_k + k_idx. The expert for that row is
 *   global_e = topk_ids[token_idx, k_idx]
 *   local_e  = expert_map[global_e]   (-1 → non-local; row skipped)
 *
 * Per-row math (mirrors `_tt_grouped_fused_mul_poly_norm` plus the caller-side
 * pre-clamp; both are folded into this single fused kernel):
 *   x = clamp(input[row], -hc, +hc)         // gate
 *   m = clamp(mul[row],   -hc, +hc)         // up
 *   rms_k = rsqrt( mean(x^k) + eps )         // k = 2, 4, 6 over hidden dim D
 *   poly  = w[le,0]*x*x*x*rms_3
 *         + w[le,1]*x*x*rms_2
 *         + w[le,2]*x*rms_1
 *         + b[le,0]
 *   out[row] = (poly * m).cast<scalar_t>()
 *
 * Non-local rows (local_e == -1) are skipped — vLLM's GEMM2 masks those rows
 * out via expert_ids, so leaving the workspace untouched is correct.
 *
 * Layout:
 *   input    : [N_total, D]    (N_total = num_tokens * top_k)
 *   mul      : [N_total, D]
 *   weight   : [E_local, 3]    fp32
 *   bias     : [E_local, 1]    fp32
 *   topk_ids : [num_tokens, top_k]   int32, global expert ids
 *   expert_map: [global_E]     int32 (or null for no-EP)
 *   out      : [N_total, D]
 *
 * One CUDA block processes one row (N_total grid blocks). Two passes:
 *   1. Compute (sum_x2, sum_x4, sum_x6) via block reduction → 3 RMS values.
 *   2. Compute poly * mul, write to `out`.
 *
 * Vectorized (width=8 for half/bf16, width=4 for fp32) loads where D % w == 0.
 */

#include "type_convert.cuh"
#include "dispatch_utils.h"
#include "cub_helpers.h"
#include "libtorch_stable/quantization/vectorization.cuh"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

namespace vllm {

template <typename scalar_t, int VEC>
__global__ void grouped_poly_norm_topk_kernel(
    scalar_t* __restrict__ out,            // [N_total, D]
    const scalar_t* __restrict__ input,    // [N_total, D]   gate
    const scalar_t* __restrict__ mul,      // [N_total, D]   up
    const float* __restrict__ weight,      // [E_local, 3]
    const float* __restrict__ bias,        // [E_local, 1]
    const int* __restrict__ topk_ids,      // [num_tokens, top_k]
    const int* __restrict__ expert_map,    // [global_E] or null
    const int hidden_size,
    const int top_k,
    const float eps,
    const float hidden_clamp) {
  const int row = blockIdx.x;
  const int token_idx = row / top_k;
  const int k_idx = row % top_k;
  const int global_expert = topk_ids[token_idx * top_k + k_idx];
  const int local_expert =
      (expert_map != nullptr) ? expert_map[global_expert] : global_expert;

  // Non-local row → skip. GEMM2 masks this row via expert_ids; leaving the
  // workspace untouched matches what vLLM's silu_and_mul + Triton GEMMs do.
  if (local_expert < 0) {
    return;
  }

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const scalar_t* input_row = input + (size_t)row * hidden_size;
  const scalar_t* mul_row = mul + (size_t)row * hidden_size;
  scalar_t* out_row = out + (size_t)row * hidden_size;

  const float w0 = weight[local_expert * 3 + 0];
  const float w1 = weight[local_expert * 3 + 1];
  const float w2 = weight[local_expert * 3 + 2];
  const float b0 = bias[local_expert];

  const bool do_clamp = (hidden_clamp > 0.0f);
  const float hc = hidden_clamp;
  const float inv_d = 1.0f / static_cast<float>(hidden_size);

  // -------- Pass 1: accumulate sum(x^2), sum(x^4), sum(x^6) on clamped x --
  float s2 = 0.0f, s4 = 0.0f, s6 = 0.0f;

  if constexpr (VEC > 1) {
    using VecT = vec_n_t<scalar_t, VEC>;
    auto* v_in = reinterpret_cast<const VecT*>(input_row);
    const int vec_count = hidden_size / VEC;
    for (int i = tid; i < vec_count; i += nthreads) {
      VecT v = v_in[i];
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        float x = static_cast<float>(v.val[j]);
        if (do_clamp) x = fmaxf(-hc, fminf(hc, x));
        float x2 = x * x;
        float x4 = x2 * x2;
        float x6 = x4 * x2;
        s2 += x2;
        s4 += x4;
        s6 += x6;
      }
    }
  } else {
    for (int i = tid; i < hidden_size; i += nthreads) {
      float x = static_cast<float>(input_row[i]);
      if (do_clamp) x = fmaxf(-hc, fminf(hc, x));
      float x2 = x * x;
      float x4 = x2 * x2;
      float x6 = x4 * x2;
      s2 += x2;
      s4 += x4;
      s6 += x6;
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_tmp;
  __shared__ float s_rms1, s_rms2, s_rms3;

  float r2 = BlockReduce(reduce_tmp).Reduce(s2, CubAddOp{}, nthreads);
  __syncthreads();
  float r4 = BlockReduce(reduce_tmp).Reduce(s4, CubAddOp{}, nthreads);
  __syncthreads();
  float r6 = BlockReduce(reduce_tmp).Reduce(s6, CubAddOp{}, nthreads);
  __syncthreads();

  if (tid == 0) {
    s_rms1 = rsqrtf(r2 * inv_d + eps);  // rsqrt(mean(x^2) + eps)
    s_rms2 = rsqrtf(r4 * inv_d + eps);  // rsqrt(mean(x^4) + eps)
    s_rms3 = rsqrtf(r6 * inv_d + eps);  // rsqrt(mean(x^6) + eps)
  }
  __syncthreads();

  const float rms1 = s_rms1, rms2 = s_rms2, rms3 = s_rms3;

  // -------- Pass 2: compute poly * mul (both clamped), write output --------
  if constexpr (VEC > 1) {
    using VecT = vec_n_t<scalar_t, VEC>;
    auto* v_in = reinterpret_cast<const VecT*>(input_row);
    auto* v_mul = reinterpret_cast<const VecT*>(mul_row);
    auto* v_out = reinterpret_cast<VecT*>(out_row);
    const int vec_count = hidden_size / VEC;
    for (int i = tid; i < vec_count; i += nthreads) {
      VecT vx = v_in[i];
      VecT vm = v_mul[i];
      VecT vo;
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        float x = static_cast<float>(vx.val[j]);
        float m = static_cast<float>(vm.val[j]);
        if (do_clamp) {
          x = fmaxf(-hc, fminf(hc, x));
          m = fmaxf(-hc, fminf(hc, m));
        }
        float x2 = x * x;
        float x3 = x2 * x;
        float poly = w0 * (x3 * rms3) + w1 * (x2 * rms2) + w2 * (x * rms1) + b0;
        vo.val[j] = static_cast<scalar_t>(poly * m);
      }
      v_out[i] = vo;
    }
  } else {
    for (int i = tid; i < hidden_size; i += nthreads) {
      float x = static_cast<float>(input_row[i]);
      float m = static_cast<float>(mul_row[i]);
      if (do_clamp) {
        x = fmaxf(-hc, fminf(hc, x));
        m = fmaxf(-hc, fminf(hc, m));
      }
      float x2 = x * x;
      float x3 = x2 * x;
      float poly = w0 * (x3 * rms3) + w1 * (x2 * rms2) + w2 * (x * rms1) + b0;
      out_row[i] = static_cast<scalar_t>(poly * m);
    }
  }
}

}  // namespace vllm

// ============================================================
// Host-side launcher
// ============================================================

torch::Tensor grouped_poly_norm_forward(
    torch::Tensor const& input,
    torch::Tensor const& mul,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    torch::Tensor const& topk_ids,
    std::optional<torch::Tensor> const& expert_map,
    int64_t top_k,
    double eps,
    double hidden_clamp) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(mul.is_contiguous(), "mul must be contiguous");
  TORCH_CHECK(input.sizes() == mul.sizes(), "input/mul shape mismatch");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [N_total, D]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 3,
              "weight must be [E_local, 3]");
  TORCH_CHECK(bias.dim() == 2 && bias.size(1) == 1,
              "bias must be [E_local, 1]");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float,
              "weight must be fp32");
  TORCH_CHECK(bias.scalar_type() == at::ScalarType::Float,
              "bias must be fp32");
  TORCH_CHECK(topk_ids.dim() == 2,
              "topk_ids must be 2D [num_tokens, top_k]");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
              "topk_ids must be int32");
  TORCH_CHECK(top_k > 0, "top_k must be > 0");
  TORCH_CHECK(topk_ids.size(1) == top_k,
              "topk_ids.size(1)=", topk_ids.size(1),
              " must equal top_k=", top_k);

  const int N_total = input.size(0);
  const int D = input.size(1);
  TORCH_CHECK(N_total == topk_ids.size(0) * top_k,
              "input rows=", N_total,
              " must equal num_tokens*top_k=",
              topk_ids.size(0) * top_k);

  auto out = torch::empty_like(input);
  if (N_total == 0) {
    return out;
  }

  const int* expert_map_ptr = nullptr;
  if (expert_map.has_value() && expert_map->defined()) {
    TORCH_CHECK(expert_map->dim() == 1, "expert_map must be 1D");
    TORCH_CHECK(expert_map->scalar_type() == at::ScalarType::Int,
                "expert_map must be int32");
    TORCH_CHECK(expert_map->is_contiguous(),
                "expert_map must be contiguous");
    expert_map_ptr = expert_map->data_ptr<int>();
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Pick block size: cap at min(D, 1024), warp-aligned.
  int block = (D < 1024) ? D : 1024;
  block = ((block + 31) / 32) * 32;
  if (block > 1024) block = 1024;
  if (block < 32) block = 32;

  // Vectorization: 8 (16B for half/bf16, 32B for fp32) → 4 → 2 → 1.
  int vec = 1;
  if (D % 8 == 0) vec = 8;
  else if (D % 4 == 0) vec = 4;
  else if (D % 2 == 0) vec = 2;

  dim3 grid(N_total);
  dim3 block_dim(block);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "grouped_poly_norm_forward", [&] {
        auto launch = [&](auto vec_const) {
          constexpr int V = decltype(vec_const)::value;
          vllm::grouped_poly_norm_topk_kernel<scalar_t, V>
              <<<grid, block_dim, 0, stream>>>(
                  out.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  mul.data_ptr<scalar_t>(),
                  weight.data_ptr<float>(),
                  bias.data_ptr<float>(),
                  topk_ids.data_ptr<int>(),
                  expert_map_ptr,
                  D, static_cast<int>(top_k),
                  static_cast<float>(eps),
                  static_cast<float>(hidden_clamp));
        };
        switch (vec) {
          case 8: launch(std::integral_constant<int, 8>{}); break;
          case 4: launch(std::integral_constant<int, 4>{}); break;
          case 2: launch(std::integral_constant<int, 2>{}); break;
          default: launch(std::integral_constant<int, 1>{}); break;
        }
      });

  return out;
}
