#include "rope.cuh"
#include "tiny_llm/cuda_utils.h"
#include <cmath>

namespace tiny_llm {
namespace kernels {

// Precompute RoPE cos/sin half cache
// cos_output: [max_seq_len, D/2]
// sin_output: [max_seq_len, D/2]
__global__ void rope_precompute_cache_kernel(float *cos_output, float *sin_output, int max_seq_len,
                                             int head_dim, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_d = head_dim / 2;
    int total = max_seq_len * half_d;

    if (idx >= total) return;

    int d = idx % half_d;   // frequency index [0, D/2)
    int pos = idx / half_d; // position [0, max_seq_len)

    // inv_freq[d] = theta^(-2d/D)
    float freq = 1.0f / powf(theta, (2.0f * static_cast<float>(d)) / static_cast<float>(head_dim));
    float angle = static_cast<float>(pos) * freq;

    cos_output[idx] = cosf(angle);
    sin_output[idx] = sinf(angle);
}

void rope_precompute_cache(float *cos_output, float *sin_output, int max_seq_len, int head_dim,
                           float theta, cudaStream_t stream) {
    if (max_seq_len <= 0 || head_dim <= 0) return;

    int half_d = head_dim / 2;
    int total = max_seq_len * half_d;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    rope_precompute_cache_kernel<<<grid_size, block_size, 0, stream>>>(
        cos_output, sin_output, max_seq_len, head_dim, theta);
}

// Apply RoPE in-place to Q and K using half-split convention
// Q: [num_tokens, Hq, D]  (token-major)
// K: [num_tokens, Hkv, D] (token-major)
// start_position 从 device 端 int 读取，使 kernel 可被 CUDA Graph 重放。
__global__ void apply_rope_inplace_kernel(half *q, half *k, const float *cos, const float *sin,
                                          int num_tokens, const int *device_start_position,
                                          int num_q_heads, int num_kv_heads, int head_dim) {
    const int start_position = *device_start_position;
    int       idx = blockIdx.x * blockDim.x + threadIdx.x;
    int       half_d = head_dim / 2;

    // Each thread handles one (token, head, half_dim_index) pair for Q
    int q_total = num_tokens * num_q_heads * half_d;
    if (idx < q_total) {
        int d = idx % half_d;
        int remaining = idx / half_d;
        int h = remaining % num_q_heads;
        int s = remaining / num_q_heads;

        int pos = start_position + s;
        int q_offset = (s * num_q_heads + h) * head_dim;

        float cos_val = cos[pos * half_d + d];
        float sin_val = sin[pos * half_d + d];

        float x1 = __half2float(q[q_offset + d]);
        float x2 = __half2float(q[q_offset + d + half_d]);

        q[q_offset + d] = __float2half(x1 * cos_val - x2 * sin_val);
        q[q_offset + d + half_d] = __float2half(x1 * sin_val + x2 * cos_val);
        return;
    }

    // K heads
    int k_total = num_tokens * num_kv_heads * half_d;
    int k_idx = idx - q_total;
    if (k_idx < k_total) {
        int d = k_idx % half_d;
        int remaining = k_idx / half_d;
        int kh = remaining % num_kv_heads;
        int s = remaining / num_kv_heads;

        int pos = start_position + s;
        int k_offset = (s * num_kv_heads + kh) * head_dim;

        float cos_val = cos[pos * half_d + d];
        float sin_val = sin[pos * half_d + d];

        float x1 = __half2float(k[k_offset + d]);
        float x2 = __half2float(k[k_offset + d + half_d]);

        k[k_offset + d] = __float2half(x1 * cos_val - x2 * sin_val);
        k[k_offset + d + half_d] = __float2half(x1 * sin_val + x2 * cos_val);
    }
}

void apply_rope_inplace(half *q, half *k, const float *cos, const float *sin, int num_tokens,
                        const int *device_start_position, int num_q_heads, int num_kv_heads,
                        int head_dim, cudaStream_t stream) {
    if (num_tokens <= 0 || head_dim <= 0 || device_start_position == nullptr) return;

    int half_d = head_dim / 2;
    int total = num_tokens * (num_q_heads + num_kv_heads) * half_d;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    apply_rope_inplace_kernel<<<grid_size, block_size, 0, stream>>>(
        q, k, cos, sin, num_tokens, device_start_position, num_q_heads, num_kv_heads, head_dim);
}

// 旧签名薄封装：把 host 端 start_position 复制到 device 后转发（测试/兼容用）。
void apply_rope_inplace(half *q, half *k, const float *cos, const float *sin, int num_tokens,
                        int start_position, int num_q_heads, int num_kv_heads, int head_dim,
                        cudaStream_t stream) {
    // thread_local：函数级 static 在多线程并发调用时有数据竞争（copyFromHost
    // 与 kernel launch 共享同一 device 缓冲）。thread_local 保持"避免每次调用
    // cudaMalloc"的初衷，同时每线程独立缓冲。
    static thread_local DeviceBuffer<int> device_pos(1);
    device_pos.copyFromHost(&start_position, 1, stream);
    apply_rope_inplace(q, k, cos, sin, num_tokens, device_pos.data(), num_q_heads, num_kv_heads,
                       head_dim, stream);
}

} // namespace kernels
} // namespace tiny_llm
