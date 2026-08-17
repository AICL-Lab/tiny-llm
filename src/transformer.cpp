#include "tiny_llm/transformer.h"
#include "attention.cuh"
#include "rope.cuh"
#include "elementwise.cuh"
#include "rmsnorm.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/logger.h"
#include "tiny_llm/validator.h"
#include "w8a16_matmul.cuh"
#include <cmath>
#include <cstdlib>

namespace tiny_llm {

void LayerWorkspace::allocate(const ModelConfig &config) {
    if (allocated) return;

    int    max_tokens = config.max_seq_len; // Support full sequence in prefill
    size_t hidden_size = static_cast<size_t>(max_tokens) * config.hidden_dim;
    size_t qkv_size = static_cast<size_t>(max_tokens) * config.num_heads * config.head_dim;
    size_t kv_size = static_cast<size_t>(max_tokens) * config.num_kv_heads * config.head_dim;
    size_t ffn_size = static_cast<size_t>(max_tokens) * config.intermediate_dim;

    CUDA_CHECK(cudaMalloc(&norm_output, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&q_buf, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&k_buf, kv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&v_buf, kv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&attn_output, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&attn_buf, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&ffn_gate, ffn_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&ffn_up, ffn_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&ffn_output, hidden_size * sizeof(half)));

    if (std::getenv("TLLM_DEBUG_ZERO")) {
        cudaMemset(norm_output, 0, hidden_size * sizeof(half));
        cudaMemset(q_buf, 0, qkv_size * sizeof(half));
        cudaMemset(k_buf, 0, kv_size * sizeof(half));
        cudaMemset(v_buf, 0, kv_size * sizeof(half));
        cudaMemset(attn_output, 0, hidden_size * sizeof(half));
        cudaMemset(attn_buf, 0, hidden_size * sizeof(half));
        cudaMemset(ffn_gate, 0, ffn_size * sizeof(half));
        cudaMemset(ffn_up, 0, ffn_size * sizeof(half));
        cudaMemset(ffn_output, 0, hidden_size * sizeof(half));
    }

    max_batch_tokens = max_tokens;
    allocated = true;
}

void LayerWorkspace::free() {
    if (!allocated) return;

    auto safe_free = [](half *&ptr) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error in LayerWorkspace::free: %s\n",
                        cudaGetErrorString(err));
            }
            ptr = nullptr;
        }
    };
    safe_free(norm_output);
    safe_free(q_buf);
    safe_free(k_buf);
    safe_free(v_buf);
    safe_free(attn_output);
    safe_free(attn_buf);
    safe_free(ffn_gate);
    safe_free(ffn_up);
    safe_free(ffn_output);

    max_batch_tokens = 0;
    allocated = false;
}

TransformerLayer::TransformerLayer(int layer_idx, const TransformerWeights &weights,
                                   const ModelConfig &config, LayerWorkspace *workspace)
    : layer_idx_(layer_idx), weights_(weights), config_(config), ws_(workspace) {}

TransformerLayer::~TransformerLayer() {} // workspace 由引擎统一管理

TransformerLayer::TransformerLayer(TransformerLayer &&other) noexcept
    : layer_idx_(other.layer_idx_), weights_(other.weights_), config_(other.config_),
      ws_(other.ws_) {
    other.ws_ = nullptr;
}

TransformerLayer &TransformerLayer::operator=(TransformerLayer &&other) noexcept {
    if (this != &other) {
        layer_idx_ = other.layer_idx_;
        ws_ = other.ws_;
        other.ws_ = nullptr;
    }
    return *this;
}

Result<void> TransformerLayer::forward(half *hidden_states, KVCacheManager &kv_cache,
                                         int seq_id, int position, const int *decode_len,
                                         const int *rope_pos, const float *rope_cos,
                                         const float *rope_sin, cudaStream_t stream) {
    // Input validation
    auto ptr_result = Validator::validateNotNull(hidden_states, "hidden_states");
    if (ptr_result.isErr()) {
        TLLM_ERROR("forward: {}", ptr_result.error());
        return ptr_result;
    }

    if (position < 0 || position >= config_.max_seq_len) {
        TLLM_ERROR("forward: invalid position {} for layer {}, max_seq_len={}", position,
                   layer_idx_, config_.max_seq_len);
        return Result<void>::err("forward: invalid position " + std::to_string(position) +
                                 " for layer " + std::to_string(layer_idx_));
    }

    if (!kv_cache.hasSequence(seq_id)) {
        TLLM_ERROR("forward: invalid seq_id {} for layer {}", seq_id, layer_idx_);
        return Result<void>::err("forward: invalid seq_id " + std::to_string(seq_id) +
                                 " for layer " + std::to_string(layer_idx_));
    }

    // Single token decode
    return runLayer(hidden_states, kv_cache, seq_id, position, 1, decode_len, rope_pos, rope_cos,
                    rope_sin, stream);
}

Result<void> TransformerLayer::forwardPrefill(half *hidden_states, KVCacheManager &kv_cache,
                                               int seq_id, int seq_len, const int *rope_pos,
                                               const float *rope_cos, const float *rope_sin,
                                               cudaStream_t stream) {
    // Input validation
    auto ptr_result = Validator::validateNotNull(hidden_states, "hidden_states");
    if (ptr_result.isErr()) {
        TLLM_ERROR("forwardPrefill: {}", ptr_result.error());
        return ptr_result;
    }

    if (seq_len <= 0) {
        TLLM_ERROR("forwardPrefill: invalid seq_len {}", seq_len);
        return Result<void>::err("forwardPrefill: invalid seq_len " + std::to_string(seq_len));
    }

    if (seq_len > config_.max_seq_len) {
        TLLM_ERROR("forwardPrefill: seq_len {} exceeds max_seq_len {}", seq_len,
                   config_.max_seq_len);
        return Result<void>::err("forwardPrefill: seq_len " + std::to_string(seq_len) +
                                 " exceeds max_seq_len " + std::to_string(config_.max_seq_len));
    }

    if (!kv_cache.hasSequence(seq_id)) {
        TLLM_ERROR("forwardPrefill: invalid seq_id {}", seq_id);
        return Result<void>::err("forwardPrefill: invalid seq_id " + std::to_string(seq_id));
    }

    // Multiple tokens prefill
    return runLayer(hidden_states, kv_cache, seq_id, 0, seq_len, nullptr, rope_pos, rope_cos,
                    rope_sin, stream);
}

Result<void> TransformerLayer::runLayer(half *hidden_states, KVCacheManager &kv_cache,
                                          int seq_id, int position, int num_tokens,
                                          const int *decode_len, const int *rope_pos,
                                          const float *rope_cos, const float *rope_sin,
                                          cudaStream_t stream) {
    const int hidden_dim = config_.hidden_dim;

    // Attention sublayer with residual: x = x + attention(rms_norm(x))
    rmsNorm(hidden_states, weights_.rms_att_weight, ws_->norm_output, num_tokens, stream);
    auto attn_result = attention(ws_->norm_output, ws_->attn_output, kv_cache, seq_id, position,
                                  num_tokens, decode_len, rope_pos, rope_cos, rope_sin, stream);
    if (attn_result.isErr()) {
        return attn_result;
    }
    kernels::add_inplace(hidden_states, ws_->attn_output, num_tokens * hidden_dim, stream);

    // FFN sublayer with residual: x = x + ffn(rms_norm(x))
    rmsNorm(hidden_states, weights_.rms_ffn_weight, ws_->norm_output, num_tokens, stream);
    feedForward(ws_->norm_output, ws_->ffn_output, num_tokens, stream);
    kernels::add_inplace(hidden_states, ws_->ffn_output, num_tokens * hidden_dim, stream);

    return Result<void>::ok();
}

Result<void> TransformerLayer::attention(const half *x, half *output, KVCacheManager &kv_cache,
                                           int seq_id, int position, int num_tokens,
                                           const int *decode_len, const int *rope_pos,
                                           const float *rope_cos, const float *rope_sin,
                                           cudaStream_t stream) {
    int hidden_dim = config_.hidden_dim;
    int num_heads = config_.num_heads;
    int num_kv_heads = config_.num_kv_heads;
    int head_dim = config_.head_dim;
    int group_size = weights_.wq.group_size;

    // Q projection: [num_tokens, hidden_dim] @ [hidden_dim, num_heads * head_dim]
    kernels::w8a16_matmul(x, weights_.wq.data, weights_.wq.scales, ws_->q_buf, num_tokens,
                          num_heads * head_dim, hidden_dim, group_size, stream);

    // K projection: [num_tokens, hidden_dim] @ [hidden_dim, num_kv_heads *
    // head_dim]
    kernels::w8a16_matmul(x, weights_.wk.data, weights_.wk.scales, ws_->k_buf, num_tokens,
                          num_kv_heads * head_dim, hidden_dim, group_size, stream);

    // V projection: [num_tokens, hidden_dim] @ [hidden_dim, num_kv_heads *
    // head_dim]
    kernels::w8a16_matmul(x, weights_.wv.data, weights_.wv.scales, ws_->v_buf, num_tokens,
                          num_kv_heads * head_dim, hidden_dim, group_size, stream);

    // Qwen2 系 attention bias：q = x@Wq^T + bq（bias 在 RoPE 之前加）
    if (weights_.wq_bias) {
        kernels::add_bias_inplace(ws_->q_buf, weights_.wq_bias, num_tokens, num_heads * head_dim,
                                  stream);
    }
    if (weights_.wk_bias) {
        kernels::add_bias_inplace(ws_->k_buf, weights_.wk_bias, num_tokens,
                                  num_kv_heads * head_dim, stream);
    }
    if (weights_.wv_bias) {
        kernels::add_bias_inplace(ws_->v_buf, weights_.wv_bias, num_tokens,
                                  num_kv_heads * head_dim, stream);
    }

    // TLLM-003: Apply RoPE to Q and K after projection, before KV append.
    // Q: [num_tokens, Hq, D], K: [num_tokens, Hkv, D]
    // 起始位置由 device int 提供（graph 重放前置）；decode 为当前 token 绝对
    // 位置，prefill 为 0。
    kernels::apply_rope_inplace(ws_->q_buf, ws_->k_buf, rope_cos, rope_sin, num_tokens, rope_pos,
                                num_heads, num_kv_heads, head_dim, stream);

    // Get KV cache pointers
    auto [k_cache, v_cache] = kv_cache.getCache(seq_id, layer_idx_);

    // Append new K, V to cache（device 写位置版本，CUDA Graph 重放前置）
    auto append_result =
        kv_cache.appendKV(seq_id, layer_idx_, ws_->k_buf, ws_->v_buf, num_tokens,
                          kv_cache.appendPosDevicePtr(), stream);
    if (append_result.isErr()) {
        TLLM_ERROR("attention: appendKV failed for layer {}: {}", layer_idx_,
                   append_result.error());
        return append_result;
    }

    // Compute attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (num_tokens == 1 && decode_len != nullptr) {
        // Decode：appendKV() writes the current token into cache but does not make it
        // visible via getSeqLen() until the caller advances once after all layers.
        // visible_len 由 device 端 int 提供（CUDA Graph 重放前置条件），
        // 其值在 decodeStep 中设为 current_seq_len + 1。
        kernels::attention_decode(ws_->q_buf, k_cache, v_cache, ws_->attn_buf, scale, num_heads,
                                  num_kv_heads, decode_len, head_dim, stream);
    } else {
        // Prefill: full attention with causal mask over the current token batch.
        // 注意：单 token prefill（seq_len==1）也会走到这里（decode_len 为
        // nullptr），此时 attention_prefill(seq_len=1) 与 attention_decode(1)
        // 数值等价（softmax 只有一个位置）。
        kernels::attention_prefill(ws_->q_buf, ws_->k_buf, ws_->v_buf, ws_->attn_buf, scale,
                                   num_heads, num_kv_heads, num_tokens, head_dim, stream);
    }

    // Output projection: 注意力输出在 attn_buf（独立缓冲，避免就地 matmul 覆盖输入）
    kernels::w8a16_matmul(ws_->attn_buf, weights_.wo.data, weights_.wo.scales, output, num_tokens,
                          hidden_dim, num_heads * head_dim, group_size, stream);

    return Result<void>::ok();
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
    kernels::w8a16_matmul(x, weights_.w1.data, weights_.w1.scales, ws_->ffn_gate, num_tokens,
                          intermediate_dim, hidden_dim, group_size, stream);

    // Up projection: [num_tokens, hidden_dim] @ [hidden_dim, intermediate_dim]
    kernels::w8a16_matmul(x, weights_.w3.data, weights_.w3.scales, ws_->ffn_up, num_tokens,
                          intermediate_dim, hidden_dim, group_size, stream);

    // SiLU activation and element-wise multiply
    kernels::silu_mul_inplace(ws_->ffn_gate, ws_->ffn_up, num_tokens * intermediate_dim, stream);

    // Down projection: [num_tokens, intermediate_dim] @ [intermediate_dim,
    // hidden_dim]
    kernels::w8a16_matmul(ws_->ffn_gate, weights_.w2.data, weights_.w2.scales, output, num_tokens,
                          hidden_dim, intermediate_dim, group_size, stream);
}

void TransformerLayer::rmsNorm(const half *x, const half *weight, half *output, int num_tokens,
                               cudaStream_t stream) {
    kernels::rmsnorm(x, weight, output, num_tokens, config_.hidden_dim, config_.rms_norm_eps,
                     stream);
}

} // namespace tiny_llm
