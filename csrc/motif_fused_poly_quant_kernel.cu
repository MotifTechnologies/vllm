// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Count-persistent fused grouped PolyNorm + per-token-group (1x128) FP8 quant
// for the motif3 DeepGEMM MoE path (padding-free standard/ag_rs route).
//
// Input layout: flat contiguous [M_sum, 2I] where M_sum = sum(align128(count_e))
// for all experts. Each expert occupies an aligned block: the first count_e rows
// are valid, the remaining (align128(count_e) - count_e) rows are padding.
//
// The kernel processes ONLY valid rows via a persistent grid-stride loop:
//   total_valid = sum(counts[0..E))
//   for w in [blockIdx.x, total_valid) step gridDim.x:
//     e = expert owning valid row w (via prefix sum upper bound)
//     flat_row = aligned_start[e] + (w - valid_prefix[e])
//     ... PolyNorm + quant math on flat_row (bit-identical to the old kernel) ...
//
// This skips padding rows entirely — the GEMM2 also skips them via psum_layout.
//
// Output (unchanged from before):
//   out_q     : [M_sum, I]        e4m3, row-major
//   out_scale : logical [M_sum, I//128] stored COLUMN-MAJOR
//               (underlying [I//128, M_sum]; elem[g,row] @ g*M_sum + row)

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "motif_fused_quant_common.cuh"

namespace vllm {

namespace {
constexpr int GROUP = 128;  // 1x128 quant group == blockDim
constexpr int MAX_EXPERTS = 256;  // max supported local experts
}  // namespace

template <typename scalar_t, bool PACKED_SCALE>
__global__ void grouped_poly_norm_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ out_q,      // [M_sum, I] row-major
    float* __restrict__ out_scale,          // fp32: [I/128, M_sum] col-major;
                                            // packed: int32 [ceil(I/128/4), M_sum]
                                            // (UE8M0 exponent bytes, DeepGEMM
                                            // SM100 1d1d SFA layout)
    const scalar_t* __restrict__ gate_up,   // [M_sum, 2*I] (gate || up)
    const float* __restrict__ weight,       // [E_local, 3]
    const float* __restrict__ bias,         // [E_local, 1]
    const int* __restrict__ counts,         // [E_local] valid rows per expert
    const int num_experts, const int M_sum, const int I,
    const float eps, const float hidden_clamp,
    const float out_mul, const bool use_ue8m0, const float quant_eps,
    const float fp8_min, const float fp8_max) {
  const int tid = threadIdx.x;

  // --- Build prefix sums in shared memory ---
  // valid_prefix[e] = sum(counts[0..e))   (exclusive prefix sum of counts)
  // aligned_start[e] = sum(align128(counts[0..e)))  (exclusive prefix of aligned)
  __shared__ int s_valid_prefix[MAX_EXPERTS + 1];
  __shared__ int s_aligned_start[MAX_EXPERTS + 1];
  __shared__ int s_total_valid;

  // Thread 0 builds the prefix sums (E_local is small, typically 8-64).
  if (tid == 0) {
    int vp = 0, as = 0;
    s_valid_prefix[0] = 0;
    s_aligned_start[0] = 0;
    for (int e = 0; e < num_experts; ++e) {
      int c = counts[e];
      vp += c;
      s_valid_prefix[e + 1] = vp;
      as += (c + 127) & ~127;  // align to 128
      s_aligned_start[e + 1] = as;
    }
    s_total_valid = vp;
  }
  __syncthreads();

  const int total_valid = s_total_valid;
  if (total_valid == 0) return;

  // Shared memory for per-row reductions (reused across iterations).
  __shared__ float sm[GROUP];
  __shared__ float s_rms1, s_rms2, s_rms3;

  // --- Persistent grid-stride loop over valid rows ---
  // cur_expert is a monotonic cursor: w only increases by gridDim.x each step,
  // so the expert index never decreases — the linear scan below advances it.
  int cur_expert = 0;
  for (int w = blockIdx.x; w < total_valid; w += gridDim.x) {
    // Find expert e such that valid_prefix[e] <= w < valid_prefix[e+1].
    // Linear scan from cur_expert (rows are monotonically assigned).
    while (cur_expert < num_experts - 1 &&
           w >= s_valid_prefix[cur_expert + 1]) {
      ++cur_expert;
    }
    const int e = cur_expert;
    const int local_row = w - s_valid_prefix[e];
    const int flat_row = s_aligned_start[e] + local_row;

    // Expert parameters
    const float w0 = weight[e * 3 + 0];
    const float w1 = weight[e * 3 + 1];
    const float w2 = weight[e * 3 + 2];
    const float b0 = bias[e];
    const bool do_clamp = (hidden_clamp > 0.0f);
    const float hc = hidden_clamp;
    const float inv_d = 1.0f / static_cast<float>(I);

    const scalar_t* g_row = gate_up + (size_t)flat_row * 2 * I;
    const scalar_t* u_row = g_row + I;
    __nv_fp8_e4m3* oq_row = out_q + (size_t)flat_row * I;

    // Pass 1: rms over the full row (clamped gate).
    float s2 = 0.0f, s4 = 0.0f, s6 = 0.0f;
    for (int i = tid; i < I; i += GROUP) {
      float x = static_cast<float>(g_row[i]);
      if (do_clamp) x = fmaxf(-hc, fminf(hc, x));
      float x2 = x * x, x4 = x2 * x2;
      s2 += x2;
      s4 += x4;
      s6 += x4 * x2;
    }
    float r2 = motif_block_reduce_sum<GROUP>(s2, sm, tid);
    float r4 = motif_block_reduce_sum<GROUP>(s4, sm, tid);
    float r6 = motif_block_reduce_sum<GROUP>(s6, sm, tid);
    if (tid == 0) {
      s_rms1 = rsqrtf(r2 * inv_d + eps);
      s_rms2 = rsqrtf(r4 * inv_d + eps);
      s_rms3 = rsqrtf(r6 * inv_d + eps);
    }
    __syncthreads();
    const float rms1 = s_rms1, rms2 = s_rms2, rms3 = s_rms3;

    // Pass 2: per 128-group, poly*mul -> post clamp/scale -> amax -> e4m3.
    const int n_groups = I / GROUP;
    uint32_t sf_word = 0;  // PACKED_SCALE: tid0 accumulates 4 exponent bytes
    for (int g = 0; g < n_groups; ++g) {
      const int i = g * GROUP + tid;
      float x = static_cast<float>(g_row[i]);
      float m = static_cast<float>(u_row[i]);
      if (do_clamp) {
        x = fmaxf(-hc, fminf(hc, x));
        m = fmaxf(-hc, fminf(hc, m));
      }
      float x2 = x * x, x3 = x2 * x;
      float poly = w0 * (x3 * rms3) + w1 * (x2 * rms2) + w2 * (x * rms1) + b0;
      float y = poly * m;
      if (do_clamp) y = fmaxf(-hc, fminf(hc, y));  // PolyNorm output clamp
      y *= out_mul;

      float amax = motif_block_reduce_max<GROUP>(fabsf(y), sm, tid);
      float scale = motif_fp8_scale(amax, fp8_max, quant_eps, use_ue8m0);
      float q = fminf(fmaxf(y / scale, fp8_min), fp8_max);
      oq_row[i] = __nv_fp8_e4m3(q);
      if (tid == 0) {
        if (PACKED_SCALE) {
          // UE8M0 scale == power of two; the fp32 exponent byte IS the
          // UE8M0 encoding. Pack 4 consecutive groups little-endian into
          // one int32 word of DeepGEMM's M-major packed SFA layout.
          sf_word |= ((__float_as_uint(scale) >> 23) & 0xFFu) << (8 * (g & 3));
          if ((g & 3) == 3 || g == n_groups - 1) {
            reinterpret_cast<uint32_t*>(
                out_scale)[(size_t)(g >> 2) * M_sum + flat_row] = sf_word;
            sf_word = 0;
          }
        } else {
          out_scale[(size_t)g * M_sum + flat_row] = scale;
        }
      }
    }
  }
}

}  // namespace vllm

// gate_up: [M_sum, 2I] (gate||up), the GEMM1 output (contiguous).
// counts: [E_local] int32, valid rows per expert.
// Returns (out_q [M_sum,I] e4m3, out_scale [M_sum, I/128] column-major).
std::tuple<torch::Tensor, torch::Tensor> grouped_poly_norm_fp8_quant(
    torch::Tensor const& gate_up, torch::Tensor const& weight,
    torch::Tensor const& bias, torch::Tensor const& counts, double eps,
    double hidden_clamp, double polynorm_output_scale, bool use_ue8m0,
    bool packed_scale) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous(),
              "gate_up must be cuda + contiguous");
  TORCH_CHECK(gate_up.dim() == 2, "gate_up must be [M_sum, 2I]");
  TORCH_CHECK(weight.scalar_type() == at::kFloat &&
                  bias.scalar_type() == at::kFloat,
              "weight/bias must be fp32");
  TORCH_CHECK(counts.scalar_type() == at::kInt && counts.dim() == 1,
              "counts must be int32 [E_local]");
  TORCH_CHECK(!packed_scale || use_ue8m0,
              "packed_scale requires use_ue8m0 (UE8M0 exponent packing)");

  const int M_sum = gate_up.size(0);
  TORCH_CHECK(gate_up.size(1) % 2 == 0, "gate_up dim1 (2I) must be even");
  const int I = gate_up.size(1) / 2;
  TORCH_CHECK(I % 128 == 0, "I=", I, " must be 128-aligned");
  const int num_experts = counts.size(0);
  TORCH_CHECK(num_experts <= vllm::MAX_EXPERTS,
              "num_experts=", num_experts,
              " exceeds MAX_EXPERTS=", vllm::MAX_EXPERTS);

  auto out_q =
      torch::empty({M_sum, I}, gate_up.options().dtype(torch::kFloat8_e4m3fn));
  // fp32:   underlying [I/128, M_sum]; returned as [M_sum, I/128] col-major.
  // packed: underlying [ceil(I/128/4), M_sum] int32 (DeepGEMM packed-UE8M0
  //         SFA layout); returned as [M_sum, ceil(I/128/4)] M-major.
  const int n_words = (I / 128 + 3) / 4;
  auto scale_under = packed_scale
      ? torch::empty({n_words, M_sum}, gate_up.options().dtype(torch::kInt32))
      : torch::empty({I / 128, M_sum}, gate_up.options().dtype(torch::kFloat32));

  if (M_sum == 0) {
    return {out_q, scale_under.permute({1, 0})};
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(gate_up));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Persistent grid: num_SMs * 16 blocks (cap at reasonable limit).
  int device;
  cudaGetDevice(&device);
  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  int grid_size = num_sms * 16;
  // Don't launch more blocks than valid rows (sum of counts).
  // Use M_sum as upper bound (>= total_valid) to avoid host-side sum.
  if (grid_size > M_sum) grid_size = M_sum;
  if (grid_size < 1) grid_size = 1;

  dim3 grid(grid_size);
  dim3 block(vllm::GROUP);
  // Shared memory: prefix arrays + reduction buffer.
  // s_valid_prefix[MAX_EXPERTS+1] + s_aligned_start[MAX_EXPERTS+1] + s_total_valid
  // are __shared__ static, so no dynamic smem needed beyond GROUP floats for
  // reductions (also static). No dynamic shared memory required.
  const float fp8_max = 448.0f, fp8_min = -448.0f, quant_eps = 1e-10f;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      gate_up.scalar_type(), "grouped_poly_norm_fp8_quant", [&] {
        float* scale_ptr = packed_scale
            ? reinterpret_cast<float*>(scale_under.data_ptr<int32_t>())
            : scale_under.data_ptr<float>();
        if (packed_scale) {
          vllm::grouped_poly_norm_fp8_quant_kernel<scalar_t, true>
              <<<grid, block, 0, stream>>>(
                  reinterpret_cast<__nv_fp8_e4m3*>(out_q.data_ptr()),
                  scale_ptr,
                  gate_up.data_ptr<scalar_t>(), weight.data_ptr<float>(),
                  bias.data_ptr<float>(), counts.data_ptr<int>(),
                  num_experts, M_sum, I,
                  static_cast<float>(eps), static_cast<float>(hidden_clamp),
                  static_cast<float>(polynorm_output_scale), use_ue8m0,
                  quant_eps, fp8_min, fp8_max);
        } else {
          vllm::grouped_poly_norm_fp8_quant_kernel<scalar_t, false>
              <<<grid, block, 0, stream>>>(
                  reinterpret_cast<__nv_fp8_e4m3*>(out_q.data_ptr()),
                  scale_ptr,
                  gate_up.data_ptr<scalar_t>(), weight.data_ptr<float>(),
                  bias.data_ptr<float>(), counts.data_ptr<int>(),
                  num_experts, M_sum, I,
                  static_cast<float>(eps), static_cast<float>(hidden_clamp),
                  static_cast<float>(polynorm_output_scale), use_ue8m0,
                  quant_eps, fp8_min, fp8_max);
        }
      });

  return {out_q, scale_under.permute({1, 0})};
}
