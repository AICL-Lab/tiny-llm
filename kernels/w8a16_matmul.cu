#include "w8a16_matmul.cuh"
#include "tiny_llm/cuda_utils.h"

namespace tiny_llm {
namespace kernels {

// Dequantization kernel
__global__ void dequantize_kernel(
    const int8_t* __restrict__ weight_int8,
    const half* __restrict__ scales,
    half* __restrict__ weight_fp16,
    int K, int N, int group_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    
    if (idx < total) {
        int k = idx / N;
        int n = idx % N;
        
        int group_idx = k / group_size;
        int scale_idx = group_idx * N + n;
        
        float w_int = static_cast<float>(weight_int8[idx]);
        float s = __half2float(scales[scale_idx]);
        weight_fp16[idx] = __float2half(w_int * s);
    }
}

void dequantize_weights(
    const int8_t* weight_int8,
    const half* scales,
    half* weight_fp16,
    int K, int N,
    int group_size,
    cudaStream_t stream
) {
    int total = K * N;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    dequantize_kernel<<<grid_size, block_size, 0, stream>>>(
        weight_int8, scales, weight_fp16, K, N, group_size
    );
}

// Simple FP16 matmul reference kernel
__global__ void fp16_matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = __float2half(sum);
    }
}

void fp16_matmul_reference(
    const half* input,
    const half* weight,
    half* output,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    fp16_matmul_kernel<<<grid, block, 0, stream>>>(input, weight, output, M, N, K);
}

// W8A16 reference kernel (simple but correct)
__global__ void w8a16_matmul_reference_kernel(
    const half* __restrict__ input,
    const int8_t* __restrict__ weight,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int M, int N, int K, int group_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            // Load input
            float a = __half2float(input[row * K + k]);
            
            // Load and dequantize weight
            int8_t w_int = weight[k * N + col];
            int group_idx = k / group_size;
            float scale = __half2float(scales[group_idx * N + col]);
            float w = static_cast<float>(w_int) * scale;
            
            sum += a * w;
        }
        
        output[row * N + col] = __float2half(sum);
    }
}

void w8a16_matmul_reference(
    const half* input,
    const int8_t* weight,
    const half* scales,
    half* output,
    int M, int N, int K,
    int group_size,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    w8a16_matmul_reference_kernel<<<grid, block, 0, stream>>>(
        input, weight, scales, output, M, N, K, group_size
    );
}

// Optimized W8A16 kernel with shared memory tiling
// Each thread block computes a TILE_M x TILE_N output tile
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int BLOCK_K = 32;

__global__ void w8a16_matmul_tiled_kernel(
    const half* __restrict__ input,
    const int8_t* __restrict__ weight,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int M, int N, int K, int group_size
) {
    // Shared memory for input tile and weight tile
    __shared__ half smem_input[TILE_M][BLOCK_K + 1];  // +1 to avoid bank conflicts
    __shared__ int8_t smem_weight[BLOCK_K][TILE_N + 4];  // +4 for alignment
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Each thread computes multiple output elements
    // Thread block is (TILE_N, TILE_M/4) = (64, 16)
    // Each thread handles 4 rows
    int global_col = bx * TILE_N + tx;
    
    // Thread-local accumulators for 4 rows
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Number of tiles along K dimension
    int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int k_start = k_tile * BLOCK_K;
        
        // Cooperative load of input tile
        // Each thread loads multiple elements
        for (int i = ty; i < TILE_M; i += blockDim.y) {
            for (int j = tx; j < BLOCK_K; j += blockDim.x) {
                int global_row = by * TILE_M + i;
                int global_k = k_start + j;
                
                if (global_row < M && global_k < K) {
                    smem_input[i][j] = input[global_row * K + global_k];
                } else {
                    smem_input[i][j] = __float2half(0.0f);
                }
            }
        }
        
        // Cooperative load of weight tile
        for (int i = ty; i < BLOCK_K; i += blockDim.y) {
            for (int j = tx; j < TILE_N; j += blockDim.x) {
                int global_k = k_start + i;
                int g_col = bx * TILE_N + j;
                
                if (global_k < K && g_col < N) {
                    smem_weight[i][j] = weight[global_k * N + g_col];
                } else {
                    smem_weight[i][j] = 0;
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial products for 4 rows per thread
        if (global_col < N) {
            #pragma unroll
            for (int k = 0; k < BLOCK_K; ++k) {
                int global_k = k_start + k;
                if (global_k < K) {
                    // Get the correct scale for this k position
                    int group_idx = global_k / group_size;
                    float scale = __half2float(scales[group_idx * N + global_col]);
                    float w = static_cast<float>(smem_weight[k][tx]) * scale;
                    
                    // Accumulate for 4 rows
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        int local_row = ty * 4 + r;
                        int global_row = by * TILE_M + local_row;
                        if (global_row < M && local_row < TILE_M) {
                            float a = __half2float(smem_input[local_row][k]);
                            acc[r] += a * w;
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output for 4 rows
    if (global_col < N) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int local_row = ty * 4 + r;
            int global_row = by * TILE_M + local_row;
            if (global_row < M) {
                output[global_row * N + global_col] = __float2half(acc[r]);
            }
        }
    }
}

void w8a16_matmul(
    const half* input,
    const int8_t* weight,
    const half* scales,
    half* output,
    int M, int N, int K,
    int group_size,
    cudaStream_t stream
) {
    // For small matrices, use reference kernel
    if (M * N < 4096) {
        w8a16_matmul_reference(input, weight, scales, output, M, N, K, group_size, stream);
        return;
    }
    
    // Use tiled kernel for larger matrices
    dim3 block(TILE_N, TILE_M / 4);  // 64 x 16 threads
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    w8a16_matmul_tiled_kernel<<<grid, block, 0, stream>>>(
        input, weight, scales, output, M, N, K, group_size
    );
}

} // namespace kernels
} // namespace tiny_llm
