#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

constexpr unsigned int FULL_MASK = 0xffffffff;

// Warp-level reduction using shuffle instructions
// Reduces a value across all threads in a warp

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__device__ __forceinline__ half warp_reduce_sum_half(half val) {
    float fval = __half2float(val);
    fval = warp_reduce_sum(fval);
    return __float2half(fval);
}

// Broadcast value from lane 0 to all lanes in warp
__device__ __forceinline__ float warp_broadcast(float val, int src_lane = 0) {
    return __shfl_sync(FULL_MASK, val, src_lane);
}

// Warp-level dot product
// Each thread holds one element, computes dot product across warp
__device__ __forceinline__ float warp_dot_product(float a, float b) {
    float product = a * b;
    return warp_reduce_sum(product);
}

// Warp-level softmax (for attention)
// Input: each thread holds one score
// Output: each thread holds corresponding softmax probability
__device__ __forceinline__ float warp_softmax(float score, int valid_lanes) {
    // Find max for numerical stability
    float max_score = score;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(FULL_MASK, max_score, offset);
        if (threadIdx.x % 32 + offset < valid_lanes) {
            max_score = fmaxf(max_score, other);
        }
    }
    max_score = __shfl_sync(FULL_MASK, max_score, 0);

    // Compute exp(score - max)
    float exp_score = (threadIdx.x % 32 < valid_lanes) ? expf(score - max_score) : 0.0f;

    // Sum of exponentials
    float sum_exp = warp_reduce_sum(exp_score);
    sum_exp = __shfl_sync(FULL_MASK, sum_exp, 0);

    // Normalize
    return exp_score / (sum_exp + 1e-6f);
}
// 注：模板版 block_reduce_sum/max 已删除——结果只落在 warp 0、不跨 warp 广播，
// 属易误用陷阱；生产路径使用 attention.cu 中带广播的 block_reduce_*_dyn。

} // namespace kernels
} // namespace tiny_llm
