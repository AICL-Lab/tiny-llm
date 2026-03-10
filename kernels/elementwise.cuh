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

} // namespace kernels
} // namespace tiny_llm
