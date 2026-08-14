#include "attention.cuh"
#include "tiny_llm/cuda_utils.h"
#include "warp_utils.cuh"
#include <cfloat>

namespace tiny_llm {
namespace kernels {

// ============================================================================
// Decode attention kernel
//
// One block per query head. Each block computes attention for one Q head
// against all cached K/V positions.
//
// Layout (token-major):
//   Q:       [1,  Hq,  D]   q(h,d)      = h*D + d
//   K_cache: [T,  Hkv, D]   k(t,kh,d)   = (t*Hkv + kh)*D + d
//   V_cache: [T,  Hkv, D]   v(t,kh,d)   = (t*Hkv + kh)*D + d
//   O:       [1,  Hq,  D]   o(h,d)      = h*D + d
//
// GQA: kv_head = q_head / group_size,  group_size = Hq / Hkv
// ============================================================================
__global__ void attention_decode_kernel(const half *__restrict__ query,
                                         const half *__restrict__ k_cache,
                                         const half *__restrict__ v_cache,
                                         half *__restrict__ output, float scale, int num_q_heads,
                                         int num_kv_heads, int visible_len, int head_dim) {
    int q_head = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;

    // Pointers for this query head and its mapped KV head
    const half *q = query + q_head * head_dim;
    const half *k = k_cache + kv_head * head_dim;       // stride between positions: Hkv*D
    const half *v = v_cache + kv_head * head_dim;
    half       *o = output + q_head * head_dim;

    int kv_stride = num_kv_heads * head_dim; // stride between consecutive positions

    // Shared memory for scores and reduction
    extern __shared__ float shared[];
    float *scores = shared;                  // [visible_len]
    float *shared_reduce = shared + visible_len; // [32]

    // Compute attention scores: Q @ K^T
    for (int pos = tid; pos < visible_len; pos += block_size) {
        float score = 0.0f;
        const half *k_pos = k + pos * kv_stride;
        for (int d = 0; d < head_dim; ++d) {
            score += __half2float(q[d]) * __half2float(k_pos[d]);
        }
        scores[pos] = score * scale;
    }
    __syncthreads();

    // Softmax: find max
    float max_score = -FLT_MAX;
    for (int pos = tid; pos < visible_len; pos += block_size) {
        max_score = fmaxf(max_score, scores[pos]);
    }

    max_score = warp_reduce_max(max_score);
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) shared_reduce[warp_id] = max_score;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    if (warp_id == 0) {
        max_score = (lane < num_warps) ? shared_reduce[lane] : -FLT_MAX;
        max_score = warp_reduce_max(max_score);
    }
    __shared__ float global_max;
    if (tid == 0) global_max = max_score;
    __syncthreads();

    // Softmax: compute exp and sum
    float sum_exp = 0.0f;
    for (int pos = tid; pos < visible_len; pos += block_size) {
        float es = expf(scores[pos] - global_max);
        scores[pos] = es;
        sum_exp += es;
    }

    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) shared_reduce[warp_id] = sum_exp;
    __syncthreads();

    if (warp_id == 0) {
        sum_exp = (lane < num_warps) ? shared_reduce[lane] : 0.0f;
        sum_exp = warp_reduce_sum(sum_exp);
    }
    __shared__ float global_sum;
    if (tid == 0) global_sum = sum_exp;
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / (global_sum + 1e-6f);
    for (int pos = tid; pos < visible_len; pos += block_size) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();

    // Compute output: softmax(scores) @ V
    for (int d = tid; d < head_dim; d += block_size) {
        float out_val = 0.0f;
        for (int pos = 0; pos < visible_len; ++pos) {
            out_val += scores[pos] * __half2float(v[pos * kv_stride + d]);
        }
        o[d] = __float2half(out_val);
    }
}

void attention_decode(const half *query, const half *k_cache, const half *v_cache, half *output,
                      float scale, int num_q_heads, int num_kv_heads, int visible_len,
                      int head_dim, cudaStream_t stream) {
    if (num_q_heads <= 0 || num_kv_heads <= 0 || visible_len <= 0 || head_dim <= 0) {
        return;
    }

    int num_blocks = num_q_heads;
    int block_size = 256;
    size_t shared_size = (visible_len + 32) * sizeof(float);

    attention_decode_kernel<<<num_blocks, block_size, shared_size, stream>>>(
        query, k_cache, v_cache, output, scale, num_q_heads, num_kv_heads, visible_len, head_dim);
}

// ============================================================================
// Prefill attention kernel with causal masking
//
// One block per (query_pos, q_head).
//
// Layout (token-major):
//   Q: [S, Hq,  D]   q(s,h,d)   = (s*Hq  + h )*D + d
//   K: [S, Hkv, D]   k(s,kh,d)  = (s*Hkv + kh)*D + d
//   V: [S, Hkv, D]   v(s,kh,d)  = (s*Hkv + kh)*D + d
//   O: [S, Hq,  D]   o(s,h,d)   = (s*Hq  + h )*D + d
//
// GQA: kv_head = q_head / group_size
// ============================================================================
__global__ void attention_prefill_kernel(const half *__restrict__ query,
                                         const half *__restrict__ key,
                                         const half *__restrict__ value,
                                         half *__restrict__ output, float scale, int num_q_heads,
                                         int num_kv_heads, int seq_len, int head_dim) {
    int query_pos = blockIdx.x;
    int q_head = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;


    int kv_stride = num_kv_heads * head_dim;  // stride between tokens in K/V

    const half *q = query + (query_pos * num_q_heads + q_head) * head_dim;
    const half *k = key + kv_head * head_dim;       // base: token 0, this kv_head
    const half *v = value + kv_head * head_dim;
    half       *o = output + (query_pos * num_q_heads + q_head) * head_dim;

    extern __shared__ float shared[];
    float *scores = shared;
    float *shared_reduce = shared + seq_len;

    // Compute attention scores with causal mask
    for (int key_pos = tid; key_pos < seq_len; key_pos += block_size) {
        if (key_pos <= query_pos) {
            float score = 0.0f;
            const half *k_pos = k + key_pos * kv_stride;
            for (int d = 0; d < head_dim; ++d) {
                score += __half2float(q[d]) * __half2float(k_pos[d]);
            }
            scores[key_pos] = score * scale;
        } else {
            scores[key_pos] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Softmax
    float max_score = -FLT_MAX;
    for (int pos = tid; pos < seq_len; pos += block_size) {
        max_score = fmaxf(max_score, scores[pos]);
    }

    max_score = warp_reduce_max(max_score);
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) shared_reduce[warp_id] = max_score;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    if (warp_id == 0) {
        max_score = (lane < num_warps) ? shared_reduce[lane] : -FLT_MAX;
        max_score = warp_reduce_max(max_score);
    }
    __shared__ float global_max;
    if (tid == 0) global_max = max_score;
    __syncthreads();

    float sum_exp = 0.0f;
    for (int pos = tid; pos < seq_len; pos += block_size) {
        float es = expf(scores[pos] - global_max);
        scores[pos] = es;
        sum_exp += es;
    }

    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) shared_reduce[warp_id] = sum_exp;
    __syncthreads();

    if (warp_id == 0) {
        sum_exp = (lane < num_warps) ? shared_reduce[lane] : 0.0f;
        sum_exp = warp_reduce_sum(sum_exp);
    }
    __shared__ float global_sum;
    if (tid == 0) global_sum = sum_exp;
    __syncthreads();

    float inv_sum = 1.0f / (global_sum + 1e-6f);
    for (int pos = tid; pos < seq_len; pos += block_size) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();

    // Output
    for (int d = tid; d < head_dim; d += block_size) {
        float out_val = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            out_val += scores[pos] * __half2float(v[pos * kv_stride + d]);
        }
        o[d] = __float2half(out_val);
    }
}

void attention_prefill(const half *query, const half *key, const half *value, half *output,
                       float scale, int num_q_heads, int num_kv_heads, int seq_len, int head_dim,
                       cudaStream_t stream) {
    if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        return;
    }

    dim3 grid(seq_len, num_q_heads);
    int block_size = 256;
    size_t shared_size = (seq_len + 32) * sizeof(float);

    attention_prefill_kernel<<<grid, block_size, shared_size, stream>>>(
        query, key, value, output, scale, num_q_heads, num_kv_heads, seq_len, head_dim);
}

// Get attention weights (for testing causal mask)
// Q: [query_len, Hq,  D]
// K: [key_len,   Hkv, D]
// weights: [query_len, Hq, key_len]  (token-major)
__global__ void get_attention_weights_kernel(const half *__restrict__ query,
                                             const half *__restrict__ key,
                                             half *__restrict__ weights, float scale,
                                             int num_q_heads, int num_kv_heads, int query_len,
                                             int key_len, int head_dim, bool apply_causal_mask) {
    int query_pos = blockIdx.x;
    int q_head = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;

    int kv_stride = num_kv_heads * head_dim;

    const half *q = query + (query_pos * num_q_heads + q_head) * head_dim;
    const half *k = key + kv_head * head_dim;
    half *w = weights + (query_pos * num_q_heads + q_head) * key_len;

    for (int key_pos = tid; key_pos < key_len; key_pos += block_size) {
        if (apply_causal_mask && key_pos > query_pos) {
            w[key_pos] = __float2half(0.0f);
        } else {
            float score = 0.0f;
            const half *k_pos = k + key_pos * kv_stride;
            for (int d = 0; d < head_dim; ++d) {
                score += __half2float(q[d]) * __half2float(k_pos[d]);
            }
            w[key_pos] = __float2half(score * scale);
        }
    }
}

void get_attention_weights(const half *query, const half *key, half *__restrict__ weights,
                           float scale, int num_q_heads, int num_kv_heads, int query_len,
                           int key_len, int head_dim, bool apply_causal_mask,
                           cudaStream_t stream) {
    if (num_q_heads <= 0 || num_kv_heads <= 0 || query_len <= 0 || key_len <= 0 || head_dim <= 0) {
        return;
    }

    dim3 grid(query_len, num_q_heads);
    int block_size = 256;

    get_attention_weights_kernel<<<grid, block_size, 0, stream>>>(
        query, key, weights, scale, num_q_heads, num_kv_heads, query_len, key_len, head_dim,
        apply_causal_mask);
}

// Simple softmax kernel
__global__ void softmax_kernel(const half *__restrict__ input, half *__restrict__ output,
                               int seq_len) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const half *x = input + row * seq_len;
    half       *y = output + row * seq_len;

    extern __shared__ float shared[];

    float max_val = -FLT_MAX;
    for (int i = tid; i < seq_len; i += block_size) {
        max_val = fmaxf(max_val, __half2float(x[i]));
    }
    max_val = warp_reduce_max(max_val);

    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) shared[warp_id] = max_val;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    if (warp_id == 0) {
        max_val = (lane < num_warps) ? shared[lane] : -FLT_MAX;
        max_val = warp_reduce_max(max_val);
    }
    __shared__ float global_max;
    if (tid == 0) global_max = max_val;
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < seq_len; i += block_size) {
        float ev = expf(__half2float(x[i]) - global_max);
        shared[i] = ev;
        sum += ev;
    }
    __syncthreads();

    sum = warp_reduce_sum(sum);
    if (lane == 0) shared[seq_len + warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < num_warps) ? shared[seq_len + lane] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    __shared__ float global_sum;
    if (tid == 0) global_sum = sum;
    __syncthreads();

    float inv_sum = 1.0f / (global_sum + 1e-6f);
    for (int i = tid; i < seq_len; i += block_size) {
        y[i] = __float2half(shared[i] * inv_sum);
    }
}

void softmax(const half *input, half *output, int batch_size, int seq_len, cudaStream_t stream) {
    if (batch_size <= 0 || seq_len <= 0) return;

    int block_size = 256;
    size_t shared_size = (seq_len + 32) * sizeof(float);

    softmax_kernel<<<batch_size, block_size, shared_size, stream>>>(input, output, seq_len);
}

} // namespace kernels
} // namespace tiny_llm
