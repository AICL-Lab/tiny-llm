#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// Input:  x      [batch, hidden_dim] or [hidden_dim]
// Weight: weight [hidden_dim]
// Output: y      [batch, hidden_dim] or [hidden_dim]

void rmsnorm(const half *__restrict__ input, const half *__restrict__ weight,
             half *__restrict__ output, int batch_size, int hidden_dim,
             float eps, cudaStream_t stream = 0);

// In-place RMSNorm (output overwrites input)
void rmsnorm_inplace(half *__restrict__ x, const half *__restrict__ weight,
                     int batch_size, int hidden_dim, float eps,
                     cudaStream_t stream = 0);

// Fused RMSNorm + residual add
// output = rmsnorm(x + residual)
void rmsnorm_residual(const half *__restrict__ input,
                      const half *__restrict__ residual,
                      const half *__restrict__ weight,
                      half *__restrict__ output, int batch_size, int hidden_dim,
                      float eps, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
