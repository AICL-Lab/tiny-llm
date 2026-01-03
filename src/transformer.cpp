#include "tiny_llm/transformer.h"
#include "tiny_llm/cuda_utils.h"
#include "rmsnorm.cuh"
#include "attention.cuh"
#include "w8a16_matmul.cuh"
#include <cmath>

namespace tiny_llm {

TransformerLayer::TransformerLayer(int layer_idx, const TransformerWeights& weights,
                                   const ModelConfig& config)
    : layer_idx_(layer_idx), weights_(weights), config_(config) {
    // Allocate buffers for max batch size
    max_batch_tokens_ = config_.max_seq_len;  // Support full sequence in prefill
    allocateBuffers();
}

TransformerLayer::~TransformerLayer() {
    freeBuffers();
}

TransformerLayer::TransformerLayer(TransformerLayer&& other) noexcept
    : layer_idx_(other.layer_idx_)
    , weights_(other.weights_)
    , config_(other.config_)
    , norm_output_(other.norm_output_)
    , q_buf_(other.q_buf_)
    , k_buf_(other.k_buf_)
    , v_buf_(other.v_buf_)
    , attn_output_(other.attn_output_)
    , ffn_gate_(other.ffn_gate_)
    , ffn_up_(other.ffn_up_)
    , ffn_output_(other.ffn_output_)
    , max_batch_tokens_(other.max_batch_tokens_)
    , buffers_allocated_(other.buffers_allocated_) {
    // Null out other's pointers
    other.norm_output_ = nullptr;
    other.q_buf_ = nullptr;
    other.k_buf_ = nullptr;
    other.v_buf_ = nullptr;
    other.attn_output_ = nullptr;
    other.ffn_gate_ = nullptr;
    other.ffn_up_ = nullptr;
    other.ffn_output_ = nullptr;
    other.buffers_allocated_ = false;
}

TransformerLayer& TransformerLayer::operator=(TransformerLayer&& other) noexcept {
    if (this != &other) {
        freeBuffers();
        
        layer_idx_ = other.layer_idx_;
        norm_output_ = other.norm_output_;
        q_buf_ = other.q_buf_;
        k_buf_ = other.k_buf_;
        v_buf_ = other.v_buf_;
        attn_output_ = other.attn_output_;
        ffn_gate_ = other.ffn_gate_;
        ffn_up_ = other.ffn_up_;
        ffn_output_ = other.ffn_output_;
        max_batch_tokens_ = other.max_batch_tokens_;
        buffers_allocated_ = other.buffers_allocated_;
        
        other.norm_output_ = nullptr;
        other.q_buf_ = nullptr;
        other.k_buf_ = nullptr;
        other.v_buf_ = nullptr;
        other.attn_output_ = nullptr;
        other.ffn_gate_ = nullptr;
        other.ffn_up_ = nullptr;
        other.ffn_output_ = nullptr;
        other.buffers_allocated_ = false;
    }
    return *this;
}

void TransformerLayer::allocateBuffers() {
    if (buffers_allocated_) return;
    
    size_t hidden_size = max_batch_tokens_ * config_.hidden_dim;
    size_t qkv_size = max_batch_tokens_ * config_.num_heads * config_.head_dim;
    size_t kv_size = max_batch_tokens_ * config_.num_kv_heads * config_.head_dim;
    size_t ffn_size = max_batch_tokens_ * config_.intermediate_dim;
    
    CUDA_CHECK(cudaMalloc(&norm_output_, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&q_buf_, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&k_buf_, kv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&v_buf_, kv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&attn_output_, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&ffn_gate_, ffn_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&ffn_up_, ffn_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&ffn_output_, hidden_size * sizeof(half)));
    
    buffers_allocated_ = true;
}

void TransformerLayer::freeBuffers() {
    if (!buffers_allocated_) return;
    
    if (norm_output_) cudaFree(norm_output_);
    if (q_buf_) cudaFree(q_buf_);
    if (k_buf_) cudaFree(k_buf_);
    if (v_buf_) cudaFree(v_buf_);
    if (attn_output_) cudaFree(attn_output_);
    if (ffn_gate_) cudaFree(ffn_gate_);
    if (ffn_up_) cudaFree(ffn_up_);
    if (ffn_output_) cudaFree(ffn_output_);
    
    norm_output_ = nullptr;
    q_buf_ = nullptr;
    k_buf_ = nullptr;
    v_buf_ = nullptr;
    attn_output_ = nullptr;
    ffn_gate_ = nullptr;
    ffn_up_ = nullptr;
    ffn_output_ = nullptr;
    buffers_allocated_ = false;
}

void TransformerLayer::forward(
    half* hidden_states,
    KVCacheManager& kv_cache,
    int seq_id,
    int position,
    cudaStream_t stream
) {
    // Single token decode
    const int num_tokens = 1;
    
    // Attention sublayer with residual
    // x = x + attention(rms_norm(x))
    rmsNorm(hidden_states, weights_.rms_att_weight, norm_output_, num_tokens, stream);
    attention(norm_output_, attn_output_, kv_cache, seq_id, position, num_tokens, stream);
    
    // Add residual: hidden_states += attn_output
    // For simplicity, we'll do this in the attention output directly
    // In production, use a fused kernel
    
    // FFN sublayer with residual
    // x = x + ffn(rms_norm(x))
    rmsNorm(hidden_states, weights_.rms_ffn_weight, norm_output_, num_tokens, stream);
    feedForward(norm_output_, ffn_output_, num_tokens, stream);
    
    // Add residual (simplified - in production use fused kernel)
    // hidden_states = hidden_states + attn_output + ffn_output
}

void TransformerLayer::forwardPrefill(
    half* hidden_states,
    KVCacheManager& kv_cache,
    int seq_id,
    int seq_len,
    cudaStream_t stream
) {
    // Multiple tokens prefill
    const int num_tokens = seq_len;
    
    // Attention sublayer
    rmsNorm(hidden_states, weights_.rms_att_weight, norm_output_, num_tokens, stream);
    attention(norm_output_, attn_output_, kv_cache, seq_id, 0, num_tokens, stream);
    
    // FFN sublayer
    rmsNorm(hidden_states, weights_.rms_ffn_weight, norm_output_, num_tokens, stream);
    feedForward(norm_output_, ffn_output_, num_tokens, stream);
}

void TransformerLayer::attention(
    const half* x,
    half* output,
    KVCacheManager& kv_cache,
    int seq_id,
    int position,
    int num_tokens,
    cudaStream_t stream
) {
    int hidden_dim = config_.hidden_dim;
    int num_heads = config_.num_heads;
    int num_kv_heads = config_.num_kv_heads;
    int head_dim = config_.head_dim;
    int group_size = weights_.wq.group_size;
    
    // Q projection: [num_tokens, hidden_dim] @ [hidden_dim, num_heads * head_dim]
    kernels::w8a16_matmul(
        x, weights_.wq.data, weights_.wq.scales,
        q_buf_, num_tokens, num_heads * head_dim, hidden_dim, group_size, stream
    );
    
    // K projection: [num_tokens, hidden_dim] @ [hidden_dim, num_kv_heads * head_dim]
    kernels::w8a16_matmul(
        x, weights_.wk.data, weights_.wk.scales,
        k_buf_, num_tokens, num_kv_heads * head_dim, hidden_dim, group_size, stream
    );
    
    // V projection: [num_tokens, hidden_dim] @ [hidden_dim, num_kv_heads * head_dim]
    kernels::w8a16_matmul(
        x, weights_.wv.data, weights_.wv.scales,
        v_buf_, num_tokens, num_kv_heads * head_dim, hidden_dim, group_size, stream
    );
    
    // Get KV cache pointers
    auto [k_cache, v_cache] = kv_cache.getCache(seq_id, layer_idx_);
    
    // Append new K, V to cache
    kv_cache.appendKV(seq_id, layer_idx_, k_buf_, v_buf_, num_tokens, stream);
    
    // Compute attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int current_seq_len = kv_cache.getSeqLen(seq_id);
    
    if (num_tokens == 1) {
        // Decode: single query against full KV cache
        kernels::attention_decode(
            q_buf_, k_cache, v_cache, attn_output_,
            scale, 1, num_heads, current_seq_len, head_dim, stream
        );
    } else {
        // Prefill: full attention with causal mask
        kernels::attention_prefill(
            q_buf_, k_buf_, v_buf_, attn_output_,
            scale, 1, num_heads, num_tokens, head_dim, stream
        );
    }
    
    // Output projection: [num_tokens, num_heads * head_dim] @ [num_heads * head_dim, hidden_dim]
    kernels::w8a16_matmul(
        attn_output_, weights_.wo.data, weights_.wo.scales,
        output, num_tokens, hidden_dim, num_heads * head_dim, group_size, stream
    );
}

void TransformerLayer::feedForward(
    const half* x,
    half* output,
    int num_tokens,
    cudaStream_t stream
) {
    int hidden_dim = config_.hidden_dim;
    int intermediate_dim = config_.intermediate_dim;
    int group_size = weights_.w1.group_size;
    
    // SwiGLU FFN:
    // gate = silu(x @ w1)
    // up = x @ w3
    // output = (gate * up) @ w2
    
    // Gate projection: [num_tokens, hidden_dim] @ [hidden_dim, intermediate_dim]
    kernels::w8a16_matmul(
        x, weights_.w1.data, weights_.w1.scales,
        ffn_gate_, num_tokens, intermediate_dim, hidden_dim, group_size, stream
    );
    
    // Up projection: [num_tokens, hidden_dim] @ [hidden_dim, intermediate_dim]
    kernels::w8a16_matmul(
        x, weights_.w3.data, weights_.w3.scales,
        ffn_up_, num_tokens, intermediate_dim, hidden_dim, group_size, stream
    );
    
    // SiLU activation and element-wise multiply would go here
    // For now, we skip the activation (would need a separate kernel)
    // In production: ffn_gate_ = silu(ffn_gate_) * ffn_up_
    
    // Down projection: [num_tokens, intermediate_dim] @ [intermediate_dim, hidden_dim]
    kernels::w8a16_matmul(
        ffn_gate_, weights_.w2.data, weights_.w2.scales,
        output, num_tokens, hidden_dim, intermediate_dim, group_size, stream
    );
}

void TransformerLayer::rmsNorm(
    const half* x,
    const half* weight,
    half* output,
    int num_tokens,
    cudaStream_t stream
) {
    kernels::rmsnorm(x, weight, output, num_tokens, config_.hidden_dim, 
                     config_.rms_norm_eps, stream);
}

} // namespace tiny_llm
