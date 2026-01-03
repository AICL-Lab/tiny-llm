#include "rmsnorm.cuh"
#include "warp_utils.cuh"
#include "tiny_llm/cuda_utils.h"

namespace tiny_llm {
namespace kernels {

// RMSNorm kernel - one block per row
// Uses warp shuffle for efficient reduction
__global__ void rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_dim,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const half* x = input + row * hidden_dim;
    half* y = output + row * hidden_dim;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }
    
    // Block-level reduction
    __shared__ float shared_mem[32];  // One per warp
    
    // Warp reduction
    sum_sq = warp_reduce_sum(sum_sq);
    
    // Write warp results to shared memory
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        shared_mem[warp_id] = sum_sq;
    }
    __syncthreads();
    
    // Final reduction in first warp
    int num_warps = (block_size + 31) / 32;
    if (warp_id == 0) {
        sum_sq = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }
    
    // Broadcast result
    __shared__ float rms_inv;
    if (tid == 0) {
        float mean_sq = sum_sq / hidden_dim;
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    
    // Apply normalization and weight
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = __half2float(x[i]);
        float w = __half2float(weight[i]);
        float normalized = val * rms_inv * w;
        y[i] = __float2half(normalized);
    }
}

void rmsnorm(
    const half* input,
    const half* weight,
    half* output,
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
) {
    // Choose block size based on hidden_dim
    int block_size = 256;
    if (hidden_dim <= 256) block_size = 128;
    if (hidden_dim <= 128) block_size = 64;
    
    rmsnorm_kernel<<<batch_size, block_size, 0, stream>>>(
        input, weight, output, hidden_dim, eps
    );
}

// In-place version
__global__ void rmsnorm_inplace_kernel(
    half* __restrict__ x,
    const half* __restrict__ weight,
    int hidden_dim,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    half* data = x + row * hidden_dim;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = __half2float(data[i]);
        sum_sq += val * val;
    }
    
    // Block-level reduction
    __shared__ float shared_mem[32];
    
    sum_sq = warp_reduce_sum(sum_sq);
    
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        shared_mem[warp_id] = sum_sq;
    }
    __syncthreads();
    
    int num_warps = (block_size + 31) / 32;
    if (warp_id == 0) {
        sum_sq = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }
    
    __shared__ float rms_inv;
    if (tid == 0) {
        float mean_sq = sum_sq / hidden_dim;
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    
    // Apply normalization and weight in-place
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = __half2float(data[i]);
        float w = __half2float(weight[i]);
        float normalized = val * rms_inv * w;
        data[i] = __float2half(normalized);
    }
}

void rmsnorm_inplace(
    half* x,
    const half* weight,
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
) {
    int block_size = 256;
    if (hidden_dim <= 256) block_size = 128;
    if (hidden_dim <= 128) block_size = 64;
    
    rmsnorm_inplace_kernel<<<batch_size, block_size, 0, stream>>>(
        x, weight, hidden_dim, eps
    );
}

// Fused RMSNorm + residual
__global__ void rmsnorm_residual_kernel(
    const half* __restrict__ input,
    const half* __restrict__ residual,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_dim,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const half* x = input + row * hidden_dim;
    const half* r = residual + row * hidden_dim;
    half* y = output + row * hidden_dim;
    
    // First pass: compute sum of squares of (x + residual)
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = __half2float(x[i]) + __half2float(r[i]);
        sum_sq += val * val;
    }
    
    // Block-level reduction
    __shared__ float shared_mem[32];
    
    sum_sq = warp_reduce_sum(sum_sq);
    
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        shared_mem[warp_id] = sum_sq;
    }
    __syncthreads();
    
    int num_warps = (block_size + 31) / 32;
    if (warp_id == 0) {
        sum_sq = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }
    
    __shared__ float rms_inv;
    if (tid == 0) {
        float mean_sq = sum_sq / hidden_dim;
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    
    // Second pass: apply normalization
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = __half2float(x[i]) + __half2float(r[i]);
        float w = __half2float(weight[i]);
        float normalized = val * rms_inv * w;
        y[i] = __float2half(normalized);
    }
}

void rmsnorm_residual(
    const half* input,
    const half* residual,
    const half* weight,
    half* output,
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
) {
    int block_size = 256;
    if (hidden_dim <= 256) block_size = 128;
    if (hidden_dim <= 128) block_size = 64;
    
    rmsnorm_residual_kernel<<<batch_size, block_size, 0, stream>>>(
        input, residual, weight, output, hidden_dim, eps
    );
}

} // namespace kernels
} // namespace tiny_llm
