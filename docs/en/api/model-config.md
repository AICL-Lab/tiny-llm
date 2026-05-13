# ModelConfig

Configuration structures for model and generation settings.

## ModelConfig

Defines model hyperparameters.

```cpp
namespace tinyllm {

struct ModelConfig {
    // Vocabulary and dimensions
    size_t vocab_size = 32000;
    size_t hidden_dim = 4096;
    size_t intermediate_dim = 11008;
    size_t head_dim = 128;

    // Layer configuration
    size_t num_layers = 32;
    size_t num_heads = 32;
    size_t num_kv_heads = 32;  // For grouped-query attention

    // Sequence configuration
    size_t max_seq_len = 2048;

    // Normalization parameters
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;

    // Validation
    bool is_valid() const;
    std::string validation_error() const;
};

}  // namespace tinyllm
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vocab_size` | `size_t` | 32000 | Size of the vocabulary |
| `hidden_dim` | `size_t` | 4096 | Hidden layer dimension |
| `intermediate_dim` | `size_t` | 11008 | FFN intermediate dimension |
| `head_dim` | `size_t` | 128 | Dimension per attention head |
| `num_layers` | `size_t` | 32 | Number of transformer layers |
| `num_heads` | `size_t` | 32 | Number of query heads |
| `num_kv_heads` | `size_t` | 32 | Number of key/value heads |
| `max_seq_len` | `size_t` | 2048 | Maximum sequence length |
| `rope_theta` | `float` | 10000.0 | RoPE frequency base |
| `rms_norm_eps` | `float` | 1e-5 | RMS normalization epsilon |

### Example

```cpp
ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

if (!config.is_valid()) {
    std::cerr << config.validation_error() << std::endl;
}
```

## GenerationConfig

Controls text generation behavior.

```cpp
namespace tinyllm {

struct GenerationConfig {
    // Token limits
    size_t max_tokens = 100;
    size_t min_tokens = 0;

    // Sampling parameters
    float temperature = 1.0f;
    float top_p = 1.0f;
    size_t top_k = 0;  // 0 = disabled

    // Repetition control
    float repetition_penalty = 1.0f;
    size_t repetition_window = 20;

    // Seed for reproducibility
    size_t seed = 0;  // 0 = random seed

    // Stop sequences
    std::vector<std::string> stop_sequences;

    // Special token IDs
    TokenId eos_token_id = 2;
    TokenId pad_token_id = 0;
};

}  // namespace tinyllm
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | `size_t` | 100 | Maximum tokens to generate |
| `min_tokens` | `size_t` | 0 | Minimum tokens before stopping |
| `temperature` | `float` | 1.0 | Sampling temperature |
| `top_p` | `float` | 1.0 | Nucleus sampling threshold |
| `top_k` | `size_t` | 0 | Top-k sampling (0 = off) |
| `repetition_penalty` | `float` | 1.0 | Penalty for repeated tokens |
| `repetition_window` | `size_t` | 20 | Window for repetition penalty |
| `seed` | `size_t` | 0 | Random seed (0 = random) |
| `stop_sequences` | `vector<string>` | {} | Sequences that stop generation |
| `eos_token_id` | `TokenId` | 2 | End-of-sequence token |
| `pad_token_id` | `TokenId` | 0 | Padding token |

### Sampling Strategies

```cpp
GenerationConfig config;

// Greedy decoding (deterministic)
config.temperature = 0.0f;

// Standard sampling
config.temperature = 0.7f;

// Top-K sampling
config.top_k = 50;
config.temperature = 0.9f;

// Nucleus (Top-P) sampling
config.top_p = 0.95f;
config.temperature = 0.9f;

// Combined (recommended)
config.top_k = 50;
config.top_p = 0.95f;
config.temperature = 0.7f;
```

## KVCacheConfig

Configures the key-value cache.

```cpp
namespace tinyllm {

struct KVCacheConfig {
    size_t max_batch_size = 1;
    size_t max_seq_len = 2048;
    bool enable_swapping = false;
    float swap_threshold = 0.9f;
};

}  // namespace tinyllm
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_batch_size` | `size_t` | 1 | Maximum concurrent sequences |
| `max_seq_len` | `size_t` | 2048 | Maximum cached sequence length |
| `enable_swapping` | `bool` | false | Enable CPU memory swapping |
| `swap_threshold` | `float` | 0.9 | GPU memory threshold for swap |

## QuantizationConfig

Configures quantization settings.

```cpp
namespace tinyllm {

enum class QuantizationType {
    INT8,
    INT4,
    FP8,
};

struct QuantizationConfig {
    QuantizationType type = QuantizationType::INT8;
    size_t group_size = 128;
    bool symmetrical = true;
};

}  // namespace tinyllm
```

## See Also

- [InferenceEngine](/en/api/inference-engine) - Main API class
- [Result\<T\>](/en/api/result) - Error handling
- [KVCacheManager](/en/api/kv-cache) - Cache management
