#include "elementwise.cuh"

namespace tiny_llm {
namespace kernels {

namespace {
__global__ void add_inplace_kernel(half *data, const half *add,
                                   int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(data[idx]);
    float b = __half2float(add[idx]);
    data[idx] = __float2half(a + b);
  }
}

__global__ void silu_mul_inplace_kernel(half *gate, const half *up,
                                        int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu = g / (1.0f + expf(-g));
    gate[idx] = __float2half(silu * u);
  }
}

__global__ void gather_embeddings_kernel(const int *tokens, const half *embedding,
                                         half *output, int num_tokens,
                                         int hidden_dim, int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * hidden_dim;
  if (idx >= total) {
    return;
  }

  int token_idx = idx / hidden_dim;
  int hidden_idx = idx % hidden_dim;
  int token_id = tokens[token_idx];

  if (token_id >= 0 && token_id < vocab_size) {
    output[idx] = embedding[token_id * hidden_dim + hidden_idx];
  } else {
    output[idx] = __float2half(0.0f);
  }
}
} // namespace

void add_inplace(half *data, const half *add, int num_elements,
                 cudaStream_t stream) {
  if (num_elements <= 0)
    return;
  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;
  add_inplace_kernel<<<grid_size, block_size, 0, stream>>>(data, add,
                                                           num_elements);
}

void silu_mul_inplace(half *gate, const half *up, int num_elements,
                      cudaStream_t stream) {
  if (num_elements <= 0)
    return;
  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;
  silu_mul_inplace_kernel<<<grid_size, block_size, 0, stream>>>(gate, up,
                                                                num_elements);
}

void gather_embeddings(const int *tokens, const half *embedding, half *output,
                       int num_tokens, int hidden_dim, int vocab_size,
                       cudaStream_t stream) {
  if (num_tokens <= 0 || hidden_dim <= 0 || !tokens || !embedding || !output) {
    return;
  }
  int total = num_tokens * hidden_dim;
  int block_size = 256;
  int grid_size = (total + block_size - 1) / block_size;
  gather_embeddings_kernel<<<grid_size, block_size, 0, stream>>>(
      tokens, embedding, output, num_tokens, hidden_dim, vocab_size);
}

} // namespace kernels
} // namespace tiny_llm
