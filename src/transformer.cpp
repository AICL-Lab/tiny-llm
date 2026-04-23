#include "tiny_llm/transformer.h"
#include "attention.cuh"
#include "elementwise.cuh"
#include "rmsnorm.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/logger.h"
#include "tiny_llm/validator.h"
#include "w8a16_matmul.cuh"
#include <cmath>

namespace tiny_llm {

TransformerLayer::TransformerLayer(int layer_idx, const TransformerWeights &weights,
                                   const ModelConfig &config)
    : layer_idx_(layer_idx), weights_(weights), config_(config) {
    // Allocate buffers for max batch size
    max_batch_tokens_ = config_.max_seq_len; // Support full sequence in prefill
    allocateBuffers();
}

TransformerLayer::~TransformerLayer() { freeBuffers(); }

TransformerLayer::TransformerLayer(TransformerLayer &&other) noexcept
    : layer_idx_(other.layer_idx_), weights_(other.weights_), config_(other.config_),
      norm_output_(other.norm_output_), q_buf_(other.q_buf_), k_buf_(other.k_buf_),
      v_buf_(other.v_buf_), attn_output_(other.attn_output_), ffn_gate_(other.ffn_gate_),
      ffn_up_(other.ffn_up_), ffn_output_(other.ffn_output_),
      max_batch_tokens_(other.max_batch_tokens_), buffers_allocated_(other.buffers_allocated_) {
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

TransformerLayer &TransformerLayer::operator=(TransformerLayer &&other) noexcept {
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

    // Use safe checks in destructor (no exceptions)
    auto safe_free = [](half *ptr) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error in freeBuffers: %s\n", cudaGetErrorString(err));
            }
        }
    };

    safe_free(norm_output_);
    safe_free(q_buf_);
    safe_free(k_buf_);
    safe_free(v_buf_);
    safe_free(attn_output_);
    safe_free(ffn_gate_);
    safe_free(ffn_up_);
    safe_free(ffn_output_);

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

void TransformerLayer::forward(half *hidden_states, KVCacheManager &kv_cache, int seq_id,
                               int position, cudaStream_t stream) {
    // Input validation
    auto ptr_result = Validator::validateNotNull(hidden_states, "hidden_states");
    if (ptr_result.isErr()) {
        TLLM_ERROR("forward: {}", ptr_result.error());
        return;
    }

    if (position < 0 || position >= config_.max_seq_len) {
        TLLM_ERROR("forward: invalid position {} for layer {}, max_seq_len={}", position,
                   layer_idx_, config_.max_seq_len);
        return;
    }

    if (!kv_cache.hasSequence(seq_id)) {
        TLLM_ERROR("forward: invalid seq_id {} for layer {}", seq_id, layer_idx_);
        return;
    }

    // Single token decode
    const int num_tokens = 1;
    const int hidden_dim = config_.hidden_dim;

    // Attention sublayer with residual
    // x = x + attention(rms_norm(x))
    rmsNorm(hidden_states, weights_.rms_att_weight, norm_output_, num_tokens, stream);
    attention(norm_output_, attn_output_, kv_cache, seq_id, position, num_tokens, stream);
    kernels::add_inplace(hidden_states, attn_output_, num_tokens * hidden_dim, stream);

    // FFN sublayer with residual
    // x = x + ffn(rms_norm(x))
    rmsNorm(hidden_states, weights_.rms_ffn_weight, norm_output_, num_tokens, stream);
    feedForward(norm_output_, ffn_output_, num_tokens, stream);
    kernels::add_inplace(hidden_states, ffn_output_, num_tokens * hidden_dim, stream);
}

void TransformerLayer::forwardPrefill(half *hidden_states, KVCacheManager &kv_cache, int seq_id,
                                      int seq_len, cudaStream_t stream) {
    // Input validation
    auto ptr_result = Validator::validateNotNull(hidden_states, "hidden_states");
    if (ptr_result.isErr()) {
        TLLM_ERROR("forwardPrefill: {}", ptr_result.error());
        return;
    }

    if (seq_len <= 0) {
        TLLM_ERROR("forwardPrefill: invalid seq_len {}", seq_len);
        return;
    }

    if (seq_len > config_.max_seq_len) {
        TLLM_ERROR("forwardPrefill: seq_len {} exceeds max_seq_len {}", seq_len,
                   config_.max_seq_len);
        return;
    }

    if (!kv_cache.hasSequence(seq_id)) {
        TLLM_ERROR("forwardPrefill: invalid seq_id {}", seq_id);
        return;
    }

    // Multiple tokens prefill
    const int num_tokens = seq_len;
    const int hidden_dim = config_.hidden_dim;

    // Attention sublayer
    rmsNorm(hidden_states, weights_.rms_att_weight, norm_output_, num_tokens, stream);
    attention(norm_output_, attn_output_, kv_cache, seq_id, 0, num_tokens, stream);
    kernels::add_inplace(hidden_states, attn_output_, num_tokens * hidden_dim, stream);

    // FFN sublayer
    rmsNorm(hidden_states, weights_.rms_ffn_weight, norm_output_, num_tokens, stream);
    feedForward(norm_output_, ffn_output_, num_tokens, stream);
    kernels::add_inplace(hidden_states, ffn_output_, num_tokens * hidden_dim, stream);
}

void TransformerLayer::attention(const half *x, half *output, KVCacheManager &kv_cache, int seq_id,
                                 int position, int num_tokens, cudaStream_t stream) {
    int hidden_dim = config_.hidden_dim;
    int num_heads = config_.num_heads;
    int num_kv_heads = config_.num_kv_heads;
    int head_dim = config_.head_dim;
    int group_size = weights_.wq.group_size;

    // Q projection: [num_tokens, hidden_dim] @ [hidden_dim, num_heads * head_dim]
    kernels::w8a16_matmul(x, weights_.wq.data, weights_.wq.scales, q_buf_, num_tokens,
                          num_heads * head_dim, hidden_dim, group_size, stream);

    // K projection: [num_tokens, hidden_dim] @ [hidden_dim, num_kv_heads *
    // head_dim]
    kernels::w8a16_matmul(x, weights_.wk.data, weights_.wk.scales, k_buf_, num_tokens,
                          num_kv_heads * head_dim, hidden_dim, group_size, stream);

    // V projection: [num_tokens, hidden_dim] @ [hidden_dim, num_kv_heads *
    // head_dim]
    kernels::w8a16_matmul(x, weights_.wv.data, weights_.wv.scales, v_buf_, num_tokens,
                          num_kv_heads * head_dim, hidden_dim, group_size, stream);

    // Get KV cache pointers
    auto [k_cache, v_cache] = kv_cache.getCache(seq_id, layer_idx_);

    // Append new K, V to cache
    kv_cache.appendKV(seq_id, layer_idx_, k_buf_, v_buf_, num_tokens, stream);

    // Compute attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int   current_seq_len = kv_cache.getSeqLen(seq_id);

    if (num_tokens == 1) {
        // Decode: appendKV() writes the current token into cache but does not make it
        // visible via getSeqLen() until the caller advances once after all layers.
        int attended_seq_len = current_seq_len + 1;
        kernels::attention_decode(q_buf_, k_cache, v_cache, attn_output_, scale, 1, num_heads,
                                  attended_seq_len, head_dim, stream);
    } else {
        // Prefill: full attention with causal mask over the current token batch.
        kernels::attention_prefill(q_buf_, k_buf_, v_buf_, attn_output_, scale, 1, num_heads,
                                   num_tokens, head_dim, stream);
    }

    // Output projection: [num_tokens, num_heads * head_dim] @ [num_heads *
    // head_dim, hidden_dim]
    kernels::w8a16_matmul(attn_output_, weights_.wo.data, weights_.wo.scales, output, num_tokens,
                          hidden_dim, num_heads * head_dim, group_size, stream);
}

void TransformerLayer::feedForward(const half *x, half *output, int num_tokens,
                                   cudaStream_t stream) {
    int hidden_dim = config_.hidden_dim;
    int intermediate_dim = config_.intermediate_dim;
    int group_size = weights_.w1.group_size;

    // SwiGLU FFN:
    // gate = silu(x @ w1)
    // up = x @ w3
    // output = (gate * up) @ w2

    // Gate projection: [num_tokens, hidden_dim] @ [hidden_dim, intermediate_dim]
    kernels::w8a16_matmul(x, weights_.w1.data, weights_.w1.scales, ffn_gate_, num_tokens,
                          intermediate_dim, hidden_dim, group_size, stream);

    // Up projection: [num_tokens, hidden_dim] @ [hidden_dim, intermediate_dim]
    kernels::w8a16_matmul(x, weights_.w3.data, weights_.w3.scales, ffn_up_, num_tokens,
                          intermediate_dim, hidden_dim, group_size, stream);

    // SiLU activation and element-wise multiply
    kernels::silu_mul_inplace(ffn_gate_, ffn_up_, num_tokens * intermediate_dim, stream);

    // Down projection: [num_tokens, intermediate_dim] @ [intermediate_dim,
    // hidden_dim]
    kernels::w8a16_matmul(ffn_gate_, weights_.w2.data, weights_.w2.scales, output, num_tokens,
                          hidden_dim, intermediate_dim, group_size, stream);
}

void TransformerLayer::rmsNorm(const half *x, const half *weight, half *output, int num_tokens,
                               cudaStream_t stream) {
    kernels::rmsnorm(x, weight, output, num_tokens, config_.hidden_dim, config_.rms_norm_eps,
                     stream);
}

} // namespace tiny_llm
