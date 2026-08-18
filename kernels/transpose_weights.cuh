#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// 转置 kernel：为 M==1 decode 快路径构建 [cols, rows] 权重布局。
//
// data: [rows, cols] -> data_t: [cols, rows]，按元素并行。
// 布局推导（与 w8a16_matmul_m1_transposed_kernel 一致）：
//   src[idx] (row-major [rows, cols])  ->  dst[src_col * rows + src_row]
void transpose_int8(const int8_t *src, int8_t *dst, int rows, int cols, cudaStream_t stream = 0);

// scales: [scale_rows, cols] -> scales_t: [cols, scale_rows]
void transpose_scales(const half *src, half *dst, int scale_rows, int cols,
                      cudaStream_t stream = 0);

// fp16 权重转置：src [rows, cols] -> dst [cols, rows]
void transpose_fp16(const half *src, half *dst, int rows, int cols, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
