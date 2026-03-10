#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// Attention computation with causal masking
// Q: [batch, num_heads, 1, head_dim] - current query
// K: [batch, num_heads, seq_len, head_dim] - cached keys
// V: [batch, num_heads, seq_len, head_dim] - cached values
// O: [batch, num_heads, 1, head_dim] - output

void attention_decode(const half *__restrict__ query,
                      const half *__restrict__ k_cache,
                      const half *__restrict__ v_cache,
                      half *__restrict__ output, float scale, int batch_size,
                      int num_heads, int seq_len, int head_dim,
                      cudaStream_t stream = 0);

// Prefill attention (full sequence)
// Q, K, V: [batch, num_heads, seq_len, head_dim]
// O: [batch, num_heads, seq_len, head_dim]
void attention_prefill(const half *__restrict__ query,
                       const half *__restrict__ key,
                       const half *__restrict__ value,
                       half *__restrict__ output, float scale, int batch_size,
                       int num_heads, int seq_len, int head_dim,
                       cudaStream_t stream = 0);

// Softmax kernel (for testing)
void softmax(const half *__restrict__ input, half *__restrict__ output,
             int batch_size, int seq_len, cudaStream_t stream = 0);

// Get attention weights for testing causal mask
void get_attention_weights(const half *__restrict__ query,
                           const half *__restrict__ key,
                           half *__restrict__ weights, float scale,
                           int batch_size, int num_heads, int query_len,
                           int key_len, int head_dim, bool apply_causal_mask,
                           cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
