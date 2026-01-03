#include "attention.cuh"
#include "warp_utils.cuh"
#include "tiny_llm/cuda_utils.h"
#include <cfloat>

namespace tiny_llm {
namespace kernels {

// Decode attention kernel - one block per (batch, head)
// Each block computes attention for one query against all keys
__global__ void attention_decode_kernel(
    const half* __restrict__ query,    // [batch, num_heads, 1, head_dim]
    const half* __restrict__ k_cache,  // [batch, num_heads, seq_len, head_dim]
    const half* __restrict__ v_cache,  // [batch, num_heads, seq_len, head_dim]
    half* __restrict__ output,         // [batch, num_heads, 1, head_dim]
    float scale,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int batch_idx = blockIdx.x / num_heads;
    int head_idx = blockIdx.x % num_heads;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Pointers for this batch and head
    const half* q = query + (batch_idx * num_heads + head_idx) * head_dim;
    const half* k = k_cache + (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const half* v = v_cache + (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    half* o = output + (batch_idx * num_heads + head_idx) * head_dim;
    
    // Shared memory for scores and intermediate results
    extern __shared__ float shared[];
    float* scores = shared;  // [seq_len]
    float* shared_reduce = shared + seq_len;  // [32] for reduction
    
    // Load query to registers (each thread loads part)
    float q_reg[8];  // Assume head_dim <= 256, each thread handles up to 8 elements
    int elems_per_thread = (head_dim + block_size - 1) / block_size;
    for (int i = 0; i < elems_per_thread && tid * elems_per_thread + i < head_dim; ++i) {
        int idx = tid * elems_per_thread + i;
        if (idx < head_dim) {
            q_reg[i] = __half2float(q[idx]);
        }
    }
    
    // Compute attention scores: Q @ K^T
    for (int pos = tid; pos < seq_len; pos += block_size) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float q_val = __half2float(q[d]);
            float k_val = __half2float(k[pos * head_dim + d]);
            score += q_val * k_val;
        }
        scores[pos] = score * scale;
    }
    __syncthreads();
    
    // Softmax: find max
    float max_score = -FLT_MAX;
    for (int pos = tid; pos < seq_len; pos += block_size) {
        max_score = fmaxf(max_score, scores[pos]);
    }
    
    // Reduce max across block
    max_score = warp_reduce_max(max_score);
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        shared_reduce[warp_id] = max_score;
    }
    __syncthreads();
    
    int num_warps = (block_size + 31) / 32;
    if (warp_id == 0) {
        max_score = (lane < num_warps) ? shared_reduce[lane] : -FLT_MAX;
        max_score = warp_reduce_max(max_score);
    }
    __shared__ float global_max;
    if (tid == 0) {
        global_max = max_score;
    }
    __syncthreads();
    
    // Softmax: compute exp and sum
    float sum_exp = 0.0f;
    for (int pos = tid; pos < seq_len; pos += block_size) {
        float exp_score = expf(scores[pos] - global_max);
        scores[pos] = exp_score;
        sum_exp += exp_score;
    }
    
    // Reduce sum
    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) {
        shared_reduce[warp_id] = sum_exp;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum_exp = (lane < num_warps) ? shared_reduce[lane] : 0.0f;
        sum_exp = warp_reduce_sum(sum_exp);
    }
    __shared__ float global_sum;
    if (tid == 0) {
        global_sum = sum_exp;
    }
    __syncthreads();
    
    // Normalize scores
    float inv_sum = 1.0f / (global_sum + 1e-6f);
    for (int pos = tid; pos < seq_len; pos += block_size) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();
    
    // Compute output: softmax(scores) @ V
    for (int d = tid; d < head_dim; d += block_size) {
        float out_val = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            float v_val = __half2float(v[pos * head_dim + d]);
            out_val += scores[pos] * v_val;
        }
        o[d] = __float2half(out_val);
    }
}

void attention_decode(
    const half* query,
    const half* k_cache,
    const half* v_cache,
    half* output,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    int num_blocks = batch_size * num_heads;
    int block_size = 256;
    
    // Shared memory: scores[seq_len] + reduce[32]
    size_t shared_size = (seq_len + 32) * sizeof(float);
    
    attention_decode_kernel<<<num_blocks, block_size, shared_size, stream>>>(
        query, k_cache, v_cache, output, scale, num_heads, seq_len, head_dim
    );
}

// Prefill attention kernel with causal masking
__global__ void attention_prefill_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    float scale,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int query_pos = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Pointers
    const half* q = query + ((batch_idx * num_heads + head_idx) * seq_len + query_pos) * head_dim;
    const half* k = key + (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const half* v = value + (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    half* o = output + ((batch_idx * num_heads + head_idx) * seq_len + query_pos) * head_dim;
    
    extern __shared__ float shared[];
    float* scores = shared;
    float* shared_reduce = shared + seq_len;
    
    // Compute attention scores with causal mask
    for (int key_pos = tid; key_pos < seq_len; key_pos += block_size) {
        if (key_pos <= query_pos) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float q_val = __half2float(q[d]);
                float k_val = __half2float(k[key_pos * head_dim + d]);
                score += q_val * k_val;
            }
            scores[key_pos] = score * scale;
        } else {
            scores[key_pos] = -FLT_MAX;  // Causal mask
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
        float exp_score = expf(scores[pos] - global_max);
        scores[pos] = exp_score;
        sum_exp += exp_score;
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
            float v_val = __half2float(v[pos * head_dim + d]);
            out_val += scores[pos] * v_val;
        }
        o[d] = __float2half(out_val);
    }
}

void attention_prefill(
    const half* query,
    const half* key,
    const half* value,
    half* output,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    dim3 grid(seq_len, num_heads, batch_size);
    int block_size = 256;
    size_t shared_size = (seq_len + 32) * sizeof(float);
    
    attention_prefill_kernel<<<grid, block_size, shared_size, stream>>>(
        query, key, value, output, scale, num_heads, seq_len, head_dim
    );
}

// Get attention weights (for testing causal mask)
__global__ void get_attention_weights_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    half* __restrict__ weights,
    float scale,
    int num_heads,
    int query_len,
    int key_len,
    int head_dim,
    bool apply_causal_mask
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int query_pos = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const half* q = query + ((batch_idx * num_heads + head_idx) * query_len + query_pos) * head_dim;
    const half* k = key + (batch_idx * num_heads + head_idx) * key_len * head_dim;
    half* w = weights + ((batch_idx * num_heads + head_idx) * query_len + query_pos) * key_len;
    
    for (int key_pos = tid; key_pos < key_len; key_pos += block_size) {
        if (apply_causal_mask && key_pos > query_pos) {
            w[key_pos] = __float2half(0.0f);
        } else {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float q_val = __half2float(q[d]);
                float k_val = __half2float(k[key_pos * head_dim + d]);
                score += q_val * k_val;
            }
            w[key_pos] = __float2half(score * scale);
        }
    }
}

void get_attention_weights(
    const half* query,
    const half* key,
    half* weights,
    float scale,
    int batch_size,
    int num_heads,
    int query_len,
    int key_len,
    int head_dim,
    bool apply_causal_mask,
    cudaStream_t stream
) {
    dim3 grid(query_len, num_heads, batch_size);
    int block_size = 256;
    
    get_attention_weights_kernel<<<grid, block_size, 0, stream>>>(
        query, key, weights, scale, num_heads, query_len, key_len, head_dim, apply_causal_mask
    );
}

// Simple softmax kernel
__global__ void softmax_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int seq_len
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const half* x = input + row * seq_len;
    half* y = output + row * seq_len;
    
    extern __shared__ float shared[];
    
    // Find max
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
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < seq_len; i += block_size) {
        float exp_val = expf(__half2float(x[i]) - global_max);
        shared[i] = exp_val;
        sum += exp_val;
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
    
    // Normalize
    float inv_sum = 1.0f / (global_sum + 1e-6f);
    for (int i = tid; i < seq_len; i += block_size) {
        y[i] = __float2half(shared[i] * inv_sum);
    }
}

void softmax(
    const half* input,
    half* output,
    int batch_size,
    int seq_len,
    cudaStream_t stream
) {
    int block_size = 256;
    size_t shared_size = (seq_len + 32) * sizeof(float);
    
    softmax_kernel<<<batch_size, block_size, shared_size, stream>>>(
        input, output, seq_len
    );
}

} // namespace kernels
} // namespace tiny_llm
