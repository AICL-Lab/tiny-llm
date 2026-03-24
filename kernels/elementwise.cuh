#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// In-place elementwise add: data[i] += add[i]
void add_inplace(half *data, const half *add, int num_elements,
                 cudaStream_t stream = 0);

// In-place SiLU + multiply: gate[i] = silu(gate[i]) * up[i]
void silu_mul_inplace(half *gate, const half *up, int num_elements,
                      cudaStream_t stream = 0);

// Embedding gather: output[token_idx, hidden_idx] = embedding[token_id, hidden_idx]
void gather_embeddings(const int *tokens, const half *embedding, half *output,
                       int num_tokens, int hidden_dim, int vocab_size,
                       cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
