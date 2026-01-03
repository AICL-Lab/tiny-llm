#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tiny_llm/types.h"
#include "tiny_llm/kv_cache.h"

namespace tiny_llm {

// TransformerLayer implements a single transformer decoder layer
// with W8A16 quantized weights and KV cache support
class TransformerLayer {
public:
    TransformerLayer(int layer_idx, const TransformerWeights& weights, 
                     const ModelConfig& config);
    ~TransformerLayer();
    
    // Non-copyable
    TransformerLayer(const TransformerLayer&) = delete;
    TransformerLayer& operator=(const TransformerLayer&) = delete;
    
    // Move constructible
    TransformerLayer(TransformerLayer&& other) noexcept;
    TransformerLayer& operator=(TransformerLayer&& other) noexcept;
    
    // Forward pass for single token (decode phase)
    // hidden_states: [batch_size, hidden_dim] - input and output
    void forward(
        half* hidden_states,
        KVCacheManager& kv_cache,
        int seq_id,
        int position,
        cudaStream_t stream = 0
    );
    
    // Forward pass for multiple tokens (prefill phase)
    // hidden_states: [batch_size, seq_len, hidden_dim] - input and output
    void forwardPrefill(
        half* hidden_states,
        KVCacheManager& kv_cache,
        int seq_id,
        int seq_len,
        cudaStream_t stream = 0
    );
    
    // Get layer index
    int getLayerIdx() const { return layer_idx_; }

private:
    // Attention sublayer
    void attention(
        const half* x,
        half* output,
        KVCacheManager& kv_cache,
        int seq_id,
        int position,
        int num_tokens,
        cudaStream_t stream
    );
    
    // Feed-forward network sublayer (SwiGLU)
    void feedForward(
        const half* x,
        half* output,
        int num_tokens,
        cudaStream_t stream
    );
    
    // RMSNorm helper
    void rmsNorm(
        const half* x,
        const half* weight,
        half* output,
        int num_tokens,
        cudaStream_t stream
    );
    
    // Allocate intermediate buffers
    void allocateBuffers();
    void freeBuffers();
    
    int layer_idx_;
    const TransformerWeights& weights_;
    const ModelConfig& config_;
    
    // Intermediate buffers (GPU memory)
    half* norm_output_ = nullptr;      // [max_batch, hidden_dim]
    half* q_buf_ = nullptr;            // [max_batch, num_heads * head_dim]
    half* k_buf_ = nullptr;            // [max_batch, num_kv_heads * head_dim]
    half* v_buf_ = nullptr;            // [max_batch, num_kv_heads * head_dim]
    half* attn_output_ = nullptr;      // [max_batch, hidden_dim]
    half* ffn_gate_ = nullptr;         // [max_batch, intermediate_dim]
    half* ffn_up_ = nullptr;           // [max_batch, intermediate_dim]
    half* ffn_output_ = nullptr;       // [max_batch, hidden_dim]
    
    // Buffer sizes
    size_t max_batch_tokens_ = 0;
    bool buffers_allocated_ = false;
};

} // namespace tiny_llm
