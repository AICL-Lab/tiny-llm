#include "transpose_weights.cuh"

namespace tiny_llm {
namespace kernels {

namespace {

// Generic elementwise transpose kernel: src is [rows, cols] row-major,
// dst becomes [cols, rows] row-major.
template <typename T>
__global__ void transpose_kernel(const T *__restrict__ src, T *__restrict__ dst, int rows,
                                 int cols) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * cols;
    if (idx >= total) return;

    const size_t src_row = idx / cols;
    const size_t src_col = idx % cols;
    dst[src_col * static_cast<size_t>(rows) + src_row] = src[idx];
}

} // namespace

void transpose_int8(const int8_t *src, int8_t *dst, int rows, int cols, cudaStream_t stream) {
    if (!src || !dst || rows <= 0 || cols <= 0) return;
    const size_t total = static_cast<size_t>(rows) * cols;
    const int    block = 256;
    const int    grid = static_cast<int>((total + block - 1) / block);
    transpose_kernel<<<grid, block, 0, stream>>>(src, dst, rows, cols);
}

void transpose_scales(const half *src, half *dst, int scale_rows, int cols, cudaStream_t stream) {
    if (!src || !dst || scale_rows <= 0 || cols <= 0) return;
    const size_t total = static_cast<size_t>(scale_rows) * cols;
    const int    block = 256;
    const int    grid = static_cast<int>((total + block - 1) / block);
    transpose_kernel<<<grid, block, 0, stream>>>(src, dst, scale_rows, cols);
}

void transpose_fp16(const half *src, half *dst, int rows, int cols, cudaStream_t stream) {
    if (!src || !dst || rows <= 0 || cols <= 0) return;
    const size_t total = static_cast<size_t>(rows) * cols;
    const int    block = 256;
    const int    grid = static_cast<int>((total + block - 1) / block);
    transpose_kernel<<<grid, block, 0, stream>>>(src, dst, rows, cols);
}

} // namespace kernels
} // namespace tiny_llm
