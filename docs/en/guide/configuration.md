# Configuration

This guide covers all configuration options for Tiny-LLM.

## Model Configuration

The `ModelConfig` struct defines model hyperparameters:

```cpp
struct ModelConfig {
    size_t vocab_size;      // Vocabulary size
    size_t hidden_dim;      // Hidden dimension
    size_t intermediate_dim; // FFN intermediate dimension
    size_t num_layers;      // Number of transformer layers
    size_t num_heads;       // Number of attention heads
    size_t num_kv_heads;    // Number of KV heads (for GQA)
    size_t head_dim;        // Head dimension
    size_t max_seq_len;     // Maximum sequence length
    float rope_theta;       // RoPE theta parameter
    float rms_norm_eps;     // RMS normalization epsilon
};
```

### Example Configuration

```cpp
// LLaMA-7B style configuration
ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.intermediate_dim = 11008;
config.num_layers = 32;
config.num_heads = 32;
config.num_kv_heads = 32;
config.head_dim = 128;
config.max_seq_len = 2048;
config.rope_theta = 10000.0f;
config.rms_norm_eps = 1e-5f;
```

## Generation Configuration

The `GenerationConfig` struct controls text generation:

```cpp
struct GenerationConfig {
    size_t max_tokens;      // Maximum tokens to generate
    float temperature;      // Sampling temperature
    float top_p;            // Nucleus sampling threshold
    size_t top_k;           // Top-k sampling
    float repetition_penalty; // Repetition penalty
    size_t seed;            // Random seed (0 = random)
};
```

### Sampling Strategies

#### Greedy Decoding

```cpp
GenerationConfig config;
config.temperature = 0.0f;  // Greedy
```

#### Temperature Sampling

```cpp
GenerationConfig config;
config.temperature = 0.7f;
```

#### Top-K Sampling

```cpp
GenerationConfig config;
config.temperature = 0.9f;
config.top_k = 50;
```

#### Nucleus (Top-P) Sampling

```cpp
GenerationConfig config;
config.temperature = 0.9f;
config.top_p = 0.95f;
```

## KV Cache Configuration

```cpp
struct KVCacheConfig {
    size_t max_batch_size;  // Maximum concurrent sequences
    size_t max_seq_len;     // Maximum sequence length
    bool enable_swapping;   // Enable CPU swapping
};
```

## Quantization Configuration

```cpp
struct QuantizationConfig {
    QuantizationType type;  // INT8, INT4, etc.
    size_t group_size;      // Per-group quantization size
    bool symmetrical;       // Symmetric vs asymmetric
};
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TINYLLM_CUDA_DEVICE` | CUDA device ID | `0` |
| `TINYLLM_CACHE_DIR` | Model cache directory | `~/.tinyllm/cache` |
| `TINYLLM_LOG_LEVEL` | Log verbosity | `INFO` |

## Next Steps

- [API Reference](/en/api/) - Complete API documentation
- [Performance Guide](/en/guide/performance) - Optimization techniques
