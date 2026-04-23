#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// Kernel configuration constants
constexpr int WARP_SIZE = 32;
constexpr int TILE_K = 32; // Reduction tile size

// W8A16 Matrix Multiplication
// Computes: output = input @ dequant(weight, scales)
// Where:
//   input:  [M, K] FP16 activations
//   weight: [K, N] INT8 quantized weights
//   scales: [K/group_size, N] FP16 scale factors
//   output: [M, N] FP16 result

void w8a16_matmul(const half *__restrict__ input,    // [M, K] FP16
                  const int8_t *__restrict__ weight, // [K, N] INT8
                  const half *__restrict__ scales,   // [K/group_size, N] FP16
                  half *__restrict__ output,         // [M, N] FP16
                  int M, int N, int K, int group_size, cudaStream_t stream = 0);

// Reference implementation for testing (slower but correct)
void w8a16_matmul_reference(const half *input, const int8_t *weight, const half *scales,
                            half *output, int M, int N, int K, int group_size,
                            cudaStream_t stream = 0);

// FP16 baseline for accuracy comparison
void fp16_matmul_reference(const half *input, const half *weight, half *output, int M, int N, int K,
                           cudaStream_t stream = 0);

// Dequantize INT8 weights to FP16 (for testing)
void dequantize_weights(const int8_t *weight_int8, const half *scales, half *weight_fp16, int K,
                        int N, int group_size, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
