#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <string>
#include <vector>

namespace tiny_llm {

// Model configuration
struct ModelConfig {
  int vocab_size = 32000;
  int hidden_dim = 4096;
  int num_layers = 32;
  int num_heads = 32;
  int num_kv_heads = 32; // For GQA support
  int head_dim = 128;
  int intermediate_dim = 11008;
  int max_seq_len = 2048;
  float rope_theta = 10000.0f;
  float rms_norm_eps = 1e-5f;
  int eos_token_id = 2;
  int bos_token_id = 1;
};

// Quantization parameters
struct QuantizationParams {
  int group_size = 128;  // Elements per scale factor
  bool symmetric = true; // Symmetric quantization (no zero point)
};

// Quantized weight tensor
struct QuantizedWeight {
  int8_t *data = nullptr; // INT8 quantized weights [rows, cols]
  half *scales = nullptr; // Scale factors [ceil(rows/group_size), cols]
  int rows = 0;
  int cols = 0;
  int group_size = 128;

  // Calculate expected scale dimensions
  int scaleRows() const { return (rows + group_size - 1) / group_size; }
  int scaleCols() const { return cols; }

  // Total elements
  size_t weightElements() const { return static_cast<size_t>(rows) * cols; }
  size_t scaleElements() const {
    return static_cast<size_t>(scaleRows()) * scaleCols();
  }

  // Memory sizes
  size_t weightBytes() const { return weightElements() * sizeof(int8_t); }
  size_t scaleBytes() const { return scaleElements() * sizeof(half); }
  size_t totalBytes() const { return weightBytes() + scaleBytes(); }

  // Validate dimensions
  bool isValid() const {
    return data != nullptr && scales != nullptr && rows > 0 && cols > 0 &&
           group_size > 0;
  }
};

// Transformer layer weights
struct TransformerWeights {
  // Attention weights
  QuantizedWeight wq; // Query projection [hidden_dim, hidden_dim]
  QuantizedWeight wk; // Key projection [hidden_dim, kv_dim]
  QuantizedWeight wv; // Value projection [hidden_dim, kv_dim]
  QuantizedWeight wo; // Output projection [hidden_dim, hidden_dim]

  // FFN weights (SwiGLU)
  QuantizedWeight w1; // Gate projection [hidden_dim, intermediate_dim]
  QuantizedWeight w2; // Down projection [intermediate_dim, hidden_dim]
  QuantizedWeight w3; // Up projection [hidden_dim, intermediate_dim]

  // Normalization weights (FP16)
  half *rms_att_weight = nullptr; // [hidden_dim]
  half *rms_ffn_weight = nullptr; // [hidden_dim]
};

// Complete model weights
struct ModelWeights {
  // Token embedding [vocab_size, hidden_dim]
  half *token_embedding = nullptr;

  // Per-layer weights
  std::vector<TransformerWeights> layers;

  // Output
  half *final_norm_weight = nullptr; // [hidden_dim]
  QuantizedWeight lm_head;           // [hidden_dim, vocab_size]
};

// Generation configuration
struct GenerationConfig {
  int max_new_tokens = 256;
  float temperature = 1.0f;
  int top_k = 50;
  float top_p = 0.9f;
  bool do_sample = false; // false = greedy
  float repetition_penalty = 1.0f;
};

// Generation statistics
struct GenerationStats {
  float prefill_time_ms = 0.0f;
  float decode_time_ms = 0.0f;
  int prompt_tokens = 0;
  int tokens_generated = 0;
  float tokens_per_second = 0.0f;
  size_t peak_memory_bytes = 0;
};

// KV Cache configuration
struct KVCacheConfig {
  int num_layers = 32;
  int num_heads = 32;
  int head_dim = 128;
  int max_seq_len = 2048;
  int max_batch_size = 1;
};

} // namespace tiny_llm
