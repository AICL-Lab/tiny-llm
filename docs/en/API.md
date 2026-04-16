---
layout: default
title: "API Reference — Tiny-LLM"
description: "Complete API reference for Tiny-LLM inference engine"
nav_order: 4
---

# API Reference

Complete API reference for Tiny-LLM inference engine.

---

## Table of Contents

- [Data Types](#data-types)
- [Core Classes](#core-classes)
- [CUDA Kernels](#cuda-kernels)
- [Error Handling](#error-handling)
- [Utilities](#utilities)

---

## Data Types

### ModelConfig

Model configuration structure defining all hyperparameters.

```cpp
#include <tiny_llm/inference_engine.h>

struct ModelConfig {
    int vocab_size = 32000;           // Vocabulary size
    int hidden_dim = 4096;            // Hidden dimension
    int num_layers = 32;              // Number of Transformer layers
    int num_heads = 32;               // Number of attention heads
    int num_kv_heads = 32;            // Number of KV heads (GQA support)
    int head_dim = 128;               // Dimension per head
    int intermediate_dim = 11008;     // FFN intermediate dimension
    int max_seq_len = 2048;           // Maximum sequence length
    float rope_theta = 10000.0f;      // RoPE base frequency
    float rms_norm_eps = 1e-5f;       // RMSNorm epsilon
    int eos_token_id = 2;             // End-of-sequence token ID
    int bos_token_id = 1;             // Beginning-of-sequence token ID
};
```

**Common Configurations**:

| Model Size | hidden_dim | num_layers | num_heads | intermediate_dim |
|------------|------------|------------|-----------|------------------|
| 7B | 4096 | 32 | 32 | 11008 |
| 13B | 5120 | 40 | 40 | 13824 |
| 70B | 8192 | 80 | 64 | 28672 |

### GenerationConfig

Text generation configuration controlling sampling behavior.

```cpp
struct GenerationConfig {
    int max_new_tokens = 256;         // Maximum tokens to generate
    float temperature = 1.0f;         // Sampling temperature
    int top_k = 50;                   // Top-k sampling cutoff
    float top_p = 0.9f;               // Top-p (nucleus) sampling threshold
    bool do_sample = false;           // Enable sampling (false = greedy)
    float repetition_penalty = 1.0f;  // Penalty for repeated tokens
};
```

**Sampling Parameters**:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `temperature` | 0.0 - 2.0 | Lower = more deterministic |
| `top_k` | 1 - vocab_size | Consider only top k tokens |
| `top_p` | 0.0 - 1.0 | Consider tokens with cumulative prob ≤ p |
| `repetition_penalty` | 1.0 - 2.0 | >1.0 penalizes repeated tokens |

### QuantizedWeight

INT8 quantized weight with FP16 scales.

```cpp
struct QuantizedWeight {
    int8_t* data;                     // INT8 weights [rows, cols]
    half* scales;                     // FP16 scales [rows/group_size, cols]
    int rows;                         // Input dimension
    int cols;                         // Output dimension
    int group_size = 128;             // Quantization group size
    
    // Helper methods
    int scaleRows() const;            // ceil(rows / group_size)
    int scaleCols() const;            // cols
    size_t weightElements() const;    // rows * cols
    size_t scaleElements() const;     // scaleRows() * cols
    size_t weightBytes() const;       // weightElements()
    size_t scaleBytes() const;        // scaleElements() * 2
    size_t totalBytes() const;        // weightBytes + scaleBytes
    bool isValid() const;             // Validate dimensions
};
```

### GenerationStats

Performance statistics from text generation.

```cpp
struct GenerationStats {
    float prefill_time_ms = 0.0f;     // Prefill phase time (ms)
    float decode_time_ms = 0.0f;      // Decode phase time (ms)
    int prompt_tokens = 0;            // Number of prompt tokens
    int tokens_generated = 0;         // Number of generated tokens
    float tokens_per_second = 0.0f;   // Generation throughput
    size_t peak_memory_bytes = 0;     // Peak GPU memory usage
};
```

---

## Core Classes

### InferenceEngine

Main inference engine class. Thread-safe for concurrent generation on different engine instances.

```cpp
#include <tiny_llm/inference_engine.h>

class InferenceEngine {
public:
    // Load model from custom binary format
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& model_path,
        const ModelConfig& config
    );
    
    // Generate completion for prompt
    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& gen_config
    );
    
    // Get generation statistics
    const GenerationStats& getStats() const;
    void resetStats();
    
    // Standalone sampling functions (stateless)
    static int sampleGreedy(
        const half* logits, int vocab_size);
    
    static int sampleTemperature(
        const half* logits, int vocab_size,
        float temperature, unsigned seed = 0);
    
    static int sampleTopK(
        const half* logits, int vocab_size,
        int k, float temperature, unsigned seed = 0);
    
    static int sampleTopP(
        const half* logits, int vocab_size,
        float p, float temperature, unsigned seed = 0);
};
```

**Usage Example**:

```cpp
// Configure model
ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

// Load model
auto result = InferenceEngine::load("model.bin", config);
if (result.isErr()) {
    std::cerr << "Error: " << result.error() << std::endl;
    return 1;
}
auto engine = std::move(result.value());

// Configure generation
GenerationConfig gen_config;
gen_config.max_new_tokens = 256;
gen_config.temperature = 0.7f;
gen_config.top_p = 0.9f;
gen_config.do_sample = true;

// Generate
std::vector<int> prompt = {1, 15043, 29892};  // "Hello,"
auto output = engine->generate(prompt, gen_config);

// Check performance
const auto& stats = engine->getStats();
std::cout << "Speed: " << stats.tokens_per_second << " tok/s" << std::endl;
```

### KVCacheManager

Efficient key-value cache management for autoregressive generation.

```cpp
#include <tiny_llm/kv_cache.h>

struct KVCacheConfig {
    int num_layers = 32;              // Number of transformer layers
    int num_heads = 32;               // Number of KV heads
    int head_dim = 128;               // Dimension per head
    int max_seq_len = 2048;           // Maximum sequence length
    int max_batch_size = 1;           // Maximum batch size
};

class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& config);
    ~KVCacheManager();
    
    // Sequence management
    Result<int> allocateSequence(int max_len);
    void releaseSequence(int seq_id);
    bool hasSequence(int seq_id) const;
    
    // Cache access for attention computation
    std::pair<half*, half*> getCache(int seq_id, int layer_idx);
    int getSeqLen(int seq_id) const;
    
    // KV append (write-only, stateless)
    void appendKV(int seq_id, int layer_idx,
                  const half* new_k, const half* new_v,
                  int num_tokens, cudaStream_t stream = 0);
    
    // Advance sequence length after all layers complete
    void advanceSeqLen(int seq_id, int num_tokens);
    
    // Memory statistics
    size_t getUsedMemory() const;
    size_t getTotalMemory() const;
    size_t getFreeMemory() const;
    int getActiveSequenceCount() const;
};
```

**Usage Pattern**:

```cpp
KVCacheConfig cache_config;
cache_config.num_layers = 32;
cache_config.num_heads = 32;
cache_config.head_dim = 128;
cache_config.max_seq_len = 2048;

KVCacheManager kv_cache(cache_config);

// Allocate sequence
auto seq_result = kv_cache.allocateSequence(1024);
if (seq_result.isErr()) {
    // Handle allocation failure
}
int seq_id = seq_result.value();

// Forward pass through layers
for (int i = 0; i < num_layers; i++) {
    layers[i]->forward(hidden_states, kv_cache, seq_id, position, stream);
}

// Advance sequence length after all layers
kv_cache.advanceSeqLen(seq_id, 1);

// Release when done
kv_cache.releaseSequence(seq_id);
```

### TransformerLayer

Single transformer layer with attention and FFN.

```cpp
#include <tiny_llm/transformer.h>

class TransformerLayer {
public:
    TransformerLayer(int layer_idx,
                     const TransformerWeights& weights,
                     const ModelConfig& config);
    
    // Single token forward (decode phase)
    void forward(half* hidden_states,
                 KVCacheManager& kv_cache,
                 int seq_id,
                 int position,
                 cudaStream_t stream = 0);
    
    // Multi-token forward (prefill phase)
    void forwardPrefill(half* hidden_states,
                        KVCacheManager& kv_cache,
                        int seq_id,
                        int seq_len,
                        cudaStream_t stream = 0);
    
    int getLayerIdx() const;
};
```

---

## CUDA Kernels

### W8A16 Matrix Multiplication

```cpp
#include <w8a16_matmul.cuh>

namespace tiny_llm::kernels {

// Main W8A16 matmul kernel
void w8a16_matmul(
    const half* input,           // [M, K] FP16
    const int8_t* weight,        // [K, N] INT8 (column-major)
    const half* scales,          // [K/group_size, N] FP16
    half* output,                // [M, N] FP16
    int M,                       // Batch size
    int N,                       // Output dimension
    int K,                       // Input dimension
    int group_size,              // Quantization group size
    cudaStream_t stream = 0
);

// Weight dequantization (for testing/reference)
void dequantize_weights(
    const int8_t* weight_int8,
    const half* scales,
    half* weight_fp16,
    int K, int N,
    int group_size,
    cudaStream_t stream = 0
);

// Reference implementation for validation
void w8a16_matmul_reference(
    const half* input,
    const int8_t* weight,
    const half* scales,
    half* output,
    int M, int N, int K,
    int group_size
);

}
```

### Attention Kernels

```cpp
#include <attention.cuh>

namespace tiny_llm::kernels {

// Decode: single query token against cached KV
void attention_decode(
    const half* query,           // [batch, num_heads, 1, head_dim]
    const half* k_cache,         // [batch, num_heads, seq_len, head_dim]
    const half* v_cache,         // [batch, num_heads, seq_len, head_dim]
    half* output,                // [batch, num_heads, 1, head_dim]
    float scale,                 // 1/sqrt(head_dim)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream = 0
);

// Prefill: all query tokens with causal mask
void attention_prefill(
    const half* query,           // [batch, num_heads, seq_len, head_dim]
    const half* key,             // [batch, num_heads, seq_len, head_dim]
    const half* value,           // [batch, num_heads, seq_len, head_dim]
    half* output,                // [batch, num_heads, seq_len, head_dim]
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream = 0
);

// Standalone softmax
void softmax(
    const half* input,           // [batch, seq_len]
    half* output,
    int batch_size,
    int seq_len,
    cudaStream_t stream = 0
);

}
```

### RMSNorm

```cpp
#include <rmsnorm.cuh>

namespace tiny_llm::kernels {

// RMSNorm: output = x / sqrt(mean(x^2) + eps) * weight
void rmsnorm(
    const half* input,           // [batch, hidden_dim]
    const half* weight,          // [hidden_dim]
    half* output,                // [batch, hidden_dim]
    int batch_size,
    int hidden_dim,
    float eps = 1e-5f,
    cudaStream_t stream = 0
);

// In-place RMSNorm
void rmsnorm_inplace(
    half* x,                     // [batch, hidden_dim] (in-place)
    const half* weight,
    int batch_size,
    int hidden_dim,
    float eps = 1e-5f,
    cudaStream_t stream = 0
);

}
```

### Elementwise Operations

```cpp
#include <elementwise.cuh>

namespace tiny_llm::kernels {

// In-place addition: data[i] += add[i]
void add_inplace(
    half* data,
    const half* add,
    int num_elements,
    cudaStream_t stream = 0
);

// SwiGLU fused: gate[i] = silu(gate[i]) * up[i]
void silu_mul_inplace(
    half* gate,
    const half* up,
    int num_elements,
    cudaStream_t stream = 0
);

// Embedding lookup
void gather_embeddings(
    const int* tokens,           // [num_tokens]
    const half* embedding,       // [vocab_size, hidden_dim]
    half* output,                // [num_tokens, hidden_dim]
    int num_tokens,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream = 0
);

}
```

---

## Error Handling

### Result<T>

Rust-inspired Result type for error handling without exceptions.

```cpp
#include <tiny_llm/result.h>

template<typename T>
class Result {
public:
    // Constructors
    static Result<T> ok(T value);
    static Result<T> err(std::string message);
    
    // State checks
    bool isOk() const;
    bool isErr() const;
    
    // Value access (throws if error)
    T& value();
    const T& value() const;
    T valueOr(T default_value) const;
    
    // Error access (throws if ok)
    const std::string& error() const;
    
    // Monadic operations
    template<typename F>
    auto map(F&& f) -> Result<decltype(f(value()))>;
    
    template<typename F>
    auto flatMap(F&& f) -> decltype(f(value()));
};
```

**Usage**:

```cpp
Result<int> parseInt(const std::string& s) {
    try {
        return Result<int>::ok(std::stoi(s));
    } catch (...) {
        return Result<int>::err("Invalid integer: " + s);
    }
}

auto result = parseInt("42");
if (result.isOk()) {
    std::cout << "Value: " << result.value() << std::endl;
} else {
    std::cerr << "Error: " << result.error() << std::endl;
}

// Or with default
int val = parseInt("abc").valueOr(0);  // val = 0
```

### CudaException

CUDA error exception with context.

```cpp
#include <tiny_llm/cuda_utils.h>

class CudaException : public std::exception {
public:
    CudaException(cudaError_t err, const char* file, int line);
    
    const char* what() const noexcept override;
    cudaError_t error() const;
    const char* file() const;
    int line() const;
};

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw CudaException(err, __FILE__, __LINE__); \
        } \
    } while(0)
```

---

## Utilities

### DeviceBuffer<T>

RAII wrapper for GPU memory.

```cpp
#include <tiny_llm/cuda_utils.h>

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer();                        // Empty buffer
    explicit DeviceBuffer(size_t count);   // Allocate count elements
    ~DeviceBuffer();                       // Automatic cleanup
    
    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Movable
    DeviceBuffer(DeviceBuffer&&) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
    
    // Data access
    T* data();
    const T* data() const;
    size_t size() const;
    size_t bytes() const;
    
    // Data transfer
    void copyFromHost(const T* src, size_t count, cudaStream_t stream = 0);
    void copyToHost(T* dst, size_t count, cudaStream_t stream = 0) const;
};
```

### CudaStream

RAII wrapper for CUDA streams.

```cpp
class CudaStream {
public:
    CudaStream();
    ~CudaStream();
    
    CudaStream(const CudaStream&) = delete;
    CudaStream(CudaStream&&) noexcept;
    
    cudaStream_t get() const;
    operator cudaStream_t() const;
    
    void synchronize();
};
```

### CudaEvent

CUDA event for timing and synchronization.

```cpp
class CudaEvent {
public:
    CudaEvent();
    ~CudaEvent();
    
    void record(cudaStream_t stream = 0);
    void synchronize();
    
    static float elapsedMs(const CudaEvent& start, const CudaEvent& end);
    
    cudaEvent_t get() const;
};
```

**Timing Example**:

```cpp
CudaEvent start, end;

start.record(stream);
kernel<<<grid, block, 0, stream>>>(...);
end.record(stream);

end.synchronize();
float ms = CudaEvent::elapsedMs(start, end);
std::cout << "Kernel time: " << ms << " ms" << std::endl;
```

### StreamPool

Pool of CUDA streams for parallel execution.

```cpp
class StreamPool {
public:
    explicit StreamPool(int num_streams = 4);
    
    cudaStream_t getStream();        // Round-robin
    cudaStream_t getStream(int idx);  // By index
    
    void synchronizeAll();
    int numStreams() const;
};
```

---

**Languages**: [English](API) | [中文](../zh/API) | [Developer →](DEVELOPER)

[← Architecture](ARCHITECTURE) | [Benchmarks →](BENCHMARKS)
