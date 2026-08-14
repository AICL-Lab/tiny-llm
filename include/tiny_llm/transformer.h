#pragma once

#include "tiny_llm/kv_cache.h"
#include "tiny_llm/result.h"
#include "rope.cuh"  // TLLM-003: RoPE cache
#include "tiny_llm/types.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {

// 共享中间激活工作区：所有 TransformerLayer 复用同一组 GPU 缓冲。
// 推理逐层串行，同一时刻只有一个层在使用这些缓冲；每层独立分配会按
// 层数线性放大显存（24 层 × ~0.9GB 必然 OOM）。
struct LayerWorkspace {
    half *norm_output = nullptr; // [max_batch, hidden_dim]
    half *q_buf = nullptr;       // [max_batch, num_heads * head_dim]
    half *k_buf = nullptr;       // [max_batch, num_kv_heads * head_dim]
    half *v_buf = nullptr;       // [max_batch, num_kv_heads * head_dim]
    half *attn_output = nullptr; // [max_batch, hidden_dim]（O 投影输出，非就地）
    half *attn_buf = nullptr;    // [max_batch, hidden_dim]（注意力输出，O 投影输入）
    half *ffn_gate = nullptr;    // [max_batch, intermediate_dim]
    half *ffn_up = nullptr;      // [max_batch, intermediate_dim]
    half *ffn_output = nullptr;  // [max_batch, hidden_dim]
    int   max_batch_tokens = 0;
    bool  allocated = false;

    void allocate(const ModelConfig &config);
    void free();
    ~LayerWorkspace() { free(); }
    LayerWorkspace() = default;
    LayerWorkspace(const LayerWorkspace &) = delete;
    LayerWorkspace &operator=(const LayerWorkspace &) = delete;
};

// TransformerLayer implements a single transformer decoder layer
// with W8A16 quantized weights and KV cache support
class TransformerLayer {
  public:
    TransformerLayer(int layer_idx, const TransformerWeights &weights, const ModelConfig &config,
                     LayerWorkspace *workspace);
    ~TransformerLayer();

    // Non-copyable
    TransformerLayer(const TransformerLayer &) = delete;
    TransformerLayer &operator=(const TransformerLayer &) = delete;

    // Move constructible
    TransformerLayer(TransformerLayer &&other) noexcept;
    TransformerLayer &operator=(TransformerLayer &&other) noexcept;

    // Forward pass for single token (decode phase)
    // hidden_states: [batch_size, hidden_dim] - input and output
    // TLLM-005: Returns Result<void> for proper error propagation
    Result<void> forward(half *hidden_states, KVCacheManager &kv_cache, int seq_id, int position,
                         const float *rope_cos, const float *rope_sin, cudaStream_t stream = 0);

    // Forward pass for multiple tokens (prefill phase)
    // hidden_states: [batch_size, seq_len, hidden_dim] - input and output
    // TLLM-005: Returns Result<void> for proper error propagation
    Result<void> forwardPrefill(half *hidden_states, KVCacheManager &kv_cache, int seq_id,
                                int seq_len, const float *rope_cos, const float *rope_sin,
                                cudaStream_t stream = 0);

    // Get layer index
    int getLayerIdx() const noexcept { return layer_idx_; }

  private:
    // Attention sublayer
    Result<void> attention(const half *x, half *output, KVCacheManager &kv_cache, int seq_id,
                           int position, int num_tokens, const float *rope_cos,
                           const float *rope_sin, cudaStream_t stream);

    // Feed-forward network sublayer (SwiGLU)
    void feedForward(const half *x, half *output, int num_tokens, cudaStream_t stream);

    // RMSNorm helper
    void rmsNorm(const half *x, const half *weight, half *output, int num_tokens,
                 cudaStream_t stream);

    // Shared attention + FFN residual block for forward/forwardPrefill
    Result<void> runLayer(half *hidden_states, KVCacheManager &kv_cache, int seq_id, int position,
                          int num_tokens, const float *rope_cos, const float *rope_sin,
                          cudaStream_t stream);

    int                       layer_idx_;
    const TransformerWeights &weights_;
    const ModelConfig        &config_;

    // 共享中间激活缓冲（非拥有，由引擎统一管理生命周期）
    LayerWorkspace *ws_ = nullptr;
};

} // namespace tiny_llm
