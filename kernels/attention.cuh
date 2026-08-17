#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// ============================================================================
// Token-major layout contract (TLLM-001/002)
//
// All attention tensors use token-major physical layout:
//   Q       [S, Hq,  D]   q(s,h,d)     = ((s * Hq  + h)  * D + d)
//   K       [S, Hkv, D]   k(s,kh,d)    = ((s * Hkv + kh) * D + d)
//   V       [S, Hkv, D]   v(s,kh,d)    = ((s * Hkv + kh) * D + d)
//   K_cache [T, Hkv, D]   cache(t,kh,d)= ((t * Hkv + kh) * D + d)
//   V_cache [T, Hkv, D]
//   O       [S, Hq,  D]
//
// GQA mapping (TLLM-002):
//   group_size = Hq / Hkv
//   kv_head(qh) = qh / group_size
//
// Pre-requisites (validated by caller):
//   Hq > 0, Hkv > 0, Hq % Hkv == 0, D > 0, D even
// ============================================================================

// Decode attention: single query token against cached K/V
// Q:       [1, Hq,  D]
// K_cache: [T, Hkv, D]   (T = visible_len, includes the just-appended token)
// V_cache: [T, Hkv, D]
// O:       [1, Hq,  D]
//
// CUDA Graph 重放前置条件（任务 3.1）：visible_len 不再作为 kernel 参数，
// 而是由 device 端 int 变量提供。graph 捕获后 host 只需更新该 device 值
// （同一 stream 上的 cudaMemcpyAsync）再重放，无需重新 capture。
// 调用方必须保证 *device_visible_len 已在同一 stream 上被写入。
void attention_decode(const half *__restrict__ query, const half *__restrict__ k_cache,
                      const half *__restrict__ v_cache, half *__restrict__ output, float scale,
                      int num_q_heads, int num_kv_heads, const int *device_visible_len,
                      int head_dim, cudaStream_t stream = 0);

// 旧签名薄封装：host int 版本，内部分配/复用 device int 后转发给上面版本。
// 供测试与不需要 graph 重放的调用方使用。
void attention_decode(const half *__restrict__ query, const half *__restrict__ k_cache,
                      const half *__restrict__ v_cache, half *__restrict__ output, float scale,
                      int num_q_heads, int num_kv_heads, int visible_len, int head_dim,
                      cudaStream_t stream = 0);

// Prefill attention: full sequence with causal masking
// Q: [S, Hq,  D]
// K: [S, Hkv, D]
// V: [S, Hkv, D]
// O: [S, Hq,  D]
void attention_prefill(const half *__restrict__ query, const half *__restrict__ key,
                       const half *__restrict__ value, half *__restrict__ output, float scale,
                       int num_q_heads, int num_kv_heads, int seq_len, int head_dim,
                       cudaStream_t stream = 0);

// Softmax kernel (for testing)
void softmax(const half *__restrict__ input, half *__restrict__ output, int batch_size, int seq_len,
             cudaStream_t stream = 0);

// Get attention weights for testing causal mask
// Q: [query_len, Hq,  D]
// K: [key_len,   Hkv, D]
// weights: [query_len, Hq, key_len]  (token-major)
void get_attention_weights(const half *__restrict__ query, const half *__restrict__ key,
                           half *__restrict__ weights, float scale, int num_q_heads,
                           int num_kv_heads, int query_len, int key_len, int head_dim,
                           bool apply_causal_mask, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
