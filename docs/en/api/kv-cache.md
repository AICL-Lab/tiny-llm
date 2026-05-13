# KVCacheManager

The `KVCacheManager` class manages key-value caches for efficient autoregressive generation.

## Overview

During autoregressive generation, the model computes key (K) and value (V) tensors for each position. Caching these tensors avoids recomputation and dramatically improves throughput.

## Class Definition

```cpp
namespace tinyllm {

class KVCacheManager {
public:
    KVCacheManager();
    KVCacheManager(const KVCacheConfig& config);
    ~KVCacheManager();

    // Initialization
    Result<void> initialize(const ModelConfig& model_config, const KVCacheConfig& cache_config);
    bool is_initialized() const;

    // Sequence management
    Result<SequenceId> create_sequence();
    Result<void> destroy_sequence(SequenceId seq_id);
    void destroy_all_sequences();

    // Cache operations
    Result<void> append(SequenceId seq_id, const Tensor& keys, const Tensor& values);
    Result<std::pair<Tensor, Tensor>> get(SequenceId seq_id) const;
    Result<void> truncate(SequenceId seq_id, size_t new_length);

    // Query operations
    size_t length(SequenceId seq_id) const;
    size_t num_sequences() const;
    size_t max_sequences() const;

    // Memory management
    size_t memory_used() const;
    size_t memory_capacity() const;
    Result<void> reserve(size_t num_sequences);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tinyllm
```

## Methods

### initialize

Initializes the cache manager.

```cpp
Result<void> initialize(const ModelConfig& model_config, const KVCacheConfig& cache_config);
```

**Parameters**:
- `model_config`: Model configuration for shape computation
- `cache_config`: Cache-specific settings

**Example**:
```cpp
ModelConfig model_cfg;
model_cfg.hidden_dim = 4096;
model_cfg.num_layers = 32;
model_cfg.num_heads = 32;

KVCacheConfig cache_cfg;
cache_cfg.max_batch_size = 8;
cache_cfg.max_seq_len = 2048;

KVCacheManager cache;
Result<void> result = cache.initialize(model_cfg, cache_cfg);
```

---

### create_sequence / destroy_sequence

Manages cache slots for individual sequences.

```cpp
Result<SequenceId> create_sequence();
Result<void> destroy_sequence(SequenceId seq_id);
```

**Example**:
```cpp
// Create a new sequence
Result<SequenceId> seq_result = cache.create_sequence();
if (!seq_result.ok()) {
    std::cerr << "Failed to create sequence: " << seq_result.error_message() << std::endl;
    return;
}

SequenceId seq_id = seq_result.value();

// Use the sequence...
// ...

// Clean up when done
cache.destroy_sequence(seq_id);
```

---

### append

Appends new key-value pairs to a sequence's cache.

```cpp
Result<void> append(SequenceId seq_id, const Tensor& keys, const Tensor& values);
```

**Parameters**:
- `seq_id`: Target sequence
- `keys`: Key tensor `[num_layers, num_heads, seq_len, head_dim]`
- `values`: Value tensor `[num_layers, num_heads, seq_len, head_dim]`

**Example**:
```cpp
// After computing attention for new tokens
Tensor new_keys = compute_keys(...);
Tensor new_values = compute_values(...);

Result<void> result = cache.append(seq_id, new_keys, new_values);
```

---

### get

Retrieves the cached key-value tensors.

```cpp
Result<std::pair<Tensor, Tensor>> get(SequenceId seq_id) const;
```

**Returns**: Pair of (keys, values) tensors

**Example**:
```cpp
auto cache_result = cache.get(seq_id);
if (cache_result.ok()) {
    auto [keys, values] = cache_result.value();
    // Use in attention computation
}
```

---

### truncate

Truncates a sequence's cache to a shorter length.

```cpp
Result<void> truncate(SequenceId seq_id, size_t new_length);
```

**Use Case**: After rejecting sampled tokens during beam search or speculative decoding.

```cpp
// Roll back to position 50
cache.truncate(seq_id, 50);
```

## Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                     KV Cache Memory                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Sequence 0: [Layer 0 K/V] [Layer 1 K/V] ... [Layer N K/V] │
│              ┌──────────────────────────────────────────┐   │
│              │ Position 0, 1, 2, ..., max_seq_len       │   │
│              └──────────────────────────────────────────┘   │
│                                                              │
│  Sequence 1: [Layer 0 K/V] [Layer 1 K/V] ... [Layer N K/V] │
│              ┌──────────────────────────────────────────┐   │
│              │ Position 0, 1, 2, ..., max_seq_len       │   │
│              └──────────────────────────────────────────┘   │
│                                                              │
│  ...                                                         │
│                                                              │
│  Sequence N: ...                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### KVCacheConfig Options

```cpp
KVCacheConfig config;
config.max_batch_size = 8;     // Support 8 concurrent sequences
config.max_seq_len = 4096;     // Support longer contexts
config.enable_swapping = true; // Enable CPU offloading
config.swap_threshold = 0.85f; // Swap when 85% GPU memory used
```

### Memory Estimation

Memory required per sequence:

```
memory_per_seq = 2 × num_layers × num_kv_heads × head_dim × max_seq_len × sizeof(float16)
```

For LLaMA-7B with 2048 context:

```
memory = 2 × 32 × 32 × 128 × 2048 × 2 bytes = 1.07 GB
```

## Best Practices

### 1. Pre-allocate for Batch Size

```cpp
cache.reserve(expected_batch_size);
```

### 2. Clean Up Sequences

```cpp
// Always destroy sequences when done
cache.destroy_sequence(seq_id);

// Or destroy all at once
cache.destroy_all_sequences();
```

### 3. Monitor Memory

```cpp
size_t used = cache.memory_used();
size_t capacity = cache.memory_capacity();
float utilization = float(used) / capacity;

if (utilization > 0.9f) {
    // Consider clearing old sequences
}
```

## See Also

- [ModelConfig](/en/api/model-config) - Configuration options
- [Performance Guide](/en/guide/performance) - Optimization tips
- [Architecture](/en/guide/architecture#kv-cache) - Architecture details
