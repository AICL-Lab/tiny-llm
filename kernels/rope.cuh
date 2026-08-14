#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// ============================================================================
// RoPE (Rotary Position Embedding) - TLLM-003
//
// Uses half-split convention:
//   x1 = x[d]           (d in [0, D/2))
//   x2 = x[d + D/2]
//   out[d]       = x1 * cos[d] - x2 * sin[d]
//   out[d + D/2] = x1 * sin[d] + x2 * cos[d]
//
// Frequency:
//   inv_freq[i] = theta^(-2i/D)   for i in [0, D/2)
//   angle[pos, i] = pos * inv_freq[i]
//   cos[pos, i] = cos(angle)
//   sin[pos, i] = sin(angle)
//
// Cache layout (half cache, FP32):
//   rope_cos: [max_seq_len, D/2]
//   rope_sin: [max_seq_len, D/2]
//
// Application:
//   Q is rotated for Hq heads: [num_tokens, Hq, D]
//   K is rotated for Hkv heads: [num_tokens, Hkv, D]
//   V is not rotated.
//
// Input/output layout (token-major, same as attention):
//   q(s, h, d) = ((s * Hq  + h)  * D + d)
//   k(s, kh, d) = ((s * Hkv + kh) * D + d)
// ============================================================================

// Precompute RoPE cos/sin half cache on device.
// cos_output: [max_seq_len, head_dim/2], FP32
// sin_output: [max_seq_len, head_dim/2], FP32
void rope_precompute_cache(float *cos_output, float *sin_output, int max_seq_len, int head_dim,
                           float theta, cudaStream_t stream = 0);

// Apply RoPE in-place to Q and K.
// q: [num_tokens, num_q_heads, head_dim]  (FP16, in-place)
// k: [num_tokens, num_kv_heads, head_dim] (FP16, in-place)
// cos: [max_seq_len, head_dim/2]  (FP32, precomputed)
// sin: [max_seq_len, head_dim/2]  (FP32, precomputed)
// start_position: absolute position of the first token in this batch
void apply_rope_inplace(half *q, half *k, const float *cos, const float *sin, int num_tokens,
                        int start_position, int num_q_heads, int num_kv_heads, int head_dim,
                        cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
