---
layout: default
title: "API Reference — Tiny-LLM"
description: "Tiny-LLM Inference Engine Public API Documentation"
nav_order: 3
---

# API Reference

Complete API documentation for the Tiny-LLM inference engine.

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

Model configuration structure.

```cpp
struct ModelConfig {
    int vocab_size = 32000;       // Vocabulary size
    int hidden_dim = 4096;        // Hidden dimension
    int num_layers = 32;          // Number of Transformer layers
    int num_heads = 32;           // Number of attention heads
    int num_kv_heads = 32;        // Number of KV heads (GQA)
    int head_dim = 128;           // Dimension per head
    int intermediate_dim = 11008; // FFN intermediate dimension
    int max_seq_len = 2048;       // Maximum sequence length
    float rope_theta = 10000.0f;  // RoPE base frequency
    float rms_norm_eps = 1e-5f;   // RMSNorm epsilon
    int eos_token_id = 2;         // End-of-sequence token ID
    int bos_token_id = 1;         // Beginning-of-sequence token ID
};
```

### GenerationConfig

Text generation configuration.

```cpp
struct GenerationConfig {
    int max_new_tokens = 256;     // Maximum tokens to generate
    float temperature = 1.0f;     // Temperature for sampling
    int top_k = 50;               // Top-k sampling parameter
    float top_p = 0.9f;           // Top-p (nucleus) sampling
    bool do_sample = false;       // Enable sampling (false = greedy)
    float repetition_penalty = 1.0f;  // Penalty for repeated tokens
};
```

### QuantizedWeight

INT8 quantized weight structure.

```cpp
struct QuantizedWeight {
    int8_t* data;     // INT8 quantized weights [rows, cols]
    half* scales;     // FP16 scale factors [ceil(rows/group_size), cols]
    int rows;
    int cols;
    int group_size;   // Quantization group size (default 128)
    
    // Methods
    int scaleRows() const;        // = ceil(rows / group_size)
    int scaleCols() const;         // = cols
    size_t weightElements() const;
    size_t scaleElements() const;
    size_t weightBytes() const;
    size_t scaleBytes() const;
    size_t totalBytes() const;
    bool isValid() const;
};
```

### GenerationStats

Generation performance statistics.

```cpp
struct GenerationStats {
    float prefill_time_ms = 0.0f;    // Prefill phase time
    float decode_time_ms = 0.0f;     // Decode phase time
    int prompt_tokens = 0;            // Number of prompt tokens
    int tokens_generated = 0;         // Number of generated tokens
    float tokens_per_second = 0.0f;  // Generation throughput
    size_t peak_memory_bytes = 0;    // Peak GPU memory usage
};
```

---

## Core Classes

### InferenceEngine

Main inference engine class.

```cpp
#include <tiny_llm/inference_engine.h>

class InferenceEngine {
public:
    // Load model from file
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& model_path,
        const ModelConfig& config
    );
    
    // Generate tokens
    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& config
    );
    
    // Statistics
    const GenerationStats& getStats() const;
    void resetStats();
    
    // Sampling functions (static, can be used standalone)
    static int sampleGreedy(const half* logits, int vocab_size);
    static int sampleTemperature(const half* logits, int vocab_size, 
                                  float temperature, unsigned seed = 0);
    static int sampleTopK(const half* logits, int vocab_size, 
                          int k, float temperature, unsigned seed = 0);
    static int sampleTopP(const half* logits, int vocab_size, 
                          float p, float temperature, unsigned seed = 0);
};
```

**Example Usage**:

```cpp
ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;

auto result = InferenceEngine::load("model.bin", config);
if (result.isErr()) {
    std::cerr << "Error: " << result.error() << std::endl;
    return 1;
}

auto engine = std::move(result.value());

GenerationConfig gen_config;
gen_config.max_new_tokens = 100;
gen_config.do_sample = true;

auto output = engine->generate({1, 2, 3}, gen_config);
```

### KVCacheManager

Key-Value cache manager for efficient incremental decoding.

```cpp
#include <tiny_llm/kv_cache.h>

struct KVCacheConfig {
    int num_layers = 32;
    int num_heads = 32;
    int head_dim = 128;
    int max_seq_len = 2048;
    int max_batch_size = 1;
};

class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& config);
    
    // Sequence management
    Result<int> allocateSequence(int max_len);
    void releaseSequence(int seq_id);
    bool hasSequence(int seq_id) const;
    
    // Cache access
    std::pair<half*, half*> getCache(int seq_id, int layer_idx);
    
    // Append KV (write-only, does not advance visible length)
    void appendKV(int seq_id, int layer_idx,
                  const half* new_k, const half* new_v, 
                  int num_tokens, cudaStream_t stream = 0);
    
    // Explicitly advance sequence length (call after all layers append)
    void advanceSeqLen(int seq_id, int num_tokens);
    
    // Queries
    int getSeqLen(int seq_id) const;
    
    // Memory statistics
    size_t getUsedMemory() const;
    size_t getTotalMemory() const;
    size_t getFreeMemory() const;
    int getActiveSequenceCount() const;
};
```

**Key Design**:
- `appendKV`: Stateless write, all layers write at `current_len` position
- `advanceSeqLen`: Explicit call to advance sequence length after all layers

### TransformerLayer

Single Transformer layer implementation.

```cpp
#include <tiny_llm/transformer.h>

class TransformerLayer {
public:
    TransformerLayer(int layer_idx, const TransformerWeights& weights, 
                     const ModelConfig& config);
    
    // Single token forward (decode)
    void forward(half* hidden_states, KVCacheManager& kv_cache,
                 int seq_id, int position, cudaStream_t stream = 0);
    
    // Multi-token forward (prefill)
    void forwardPrefill(half* hidden_states, KVCacheManager& kv_cache,
                        int seq_id, int seq_len, cudaStream_t stream = 0);
    
    int getLayerIdx() const;
};
```

---

## CUDA Kernels

### W8A16 Matrix Multiplication

```cpp
#include <w8a16_matmul.cuh>

namespace tiny_llm::kernels {

// W8A16 quantized matrix multiplication
// output = input @ dequantize(weight, scales)
void w8a16_matmul(
    const half* input,      // [M, K] FP16
    const int8_t* weight,   // [K, N] INT8
    const half* scales,     // [ceil(K/group_size), N] FP16
    half* output,           // [M, N] FP16
    int M, int N, int K,
    int group_size,
    cudaStream_t stream = 0
);

// Reference implementation (for testing)
void w8a16_matmul_reference(...);

// Weight dequantization
void dequantize_weights(
    const int8_t* weight_int8,
    const half* scales,
    half* weight_fp16,
    int K, int N,
    int group_size,
    cudaStream_t stream = 0
);

}
```

### Attention

```cpp
#include <attention.cuh>

namespace tiny_llm::kernels {

// Decode attention (single token vs cached K/V)
void attention_decode(
    const half* query,      // [batch, num_heads, 1, head_dim]
    const half* k_cache,    // [batch, num_heads, seq_len, head_dim]
    const half* v_cache,    // [batch, num_heads, seq_len, head_dim]
    half* output,           // [batch, num_heads, 1, head_dim]
    float scale,            // = 1/sqrt(head_dim)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream = 0
);

// Prefill attention (multi-token, with causal mask)
void attention_prefill(
    const half* query,      // [batch, num_heads, seq_len, head_dim]
    const half* key,
    const half* value,
    half* output,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream = 0
);

// Softmax
void softmax(
    const half* input,
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

void rmsnorm(
    const half* input,      // [batch, hidden_dim]
    const half* weight,     // [hidden_dim]
    half* output,           // [batch, hidden_dim]
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream = 0
);

void rmsnorm_inplace(
    half* x,
    const half* weight,
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream = 0
);

}
```

### Elementwise Operations

```cpp
#include <elementwise.cuh>

namespace tiny_llm::kernels {

// In-place addition: data[i] += add[i]
void add_inplace(half* data, const half* add, int num_elements,
                 cudaStream_t stream = 0);

// SiLU × multiply: gate[i] = silu(gate[i]) * up[i]
void silu_mul_inplace(half* gate, const half* up, int num_elements,
                      cudaStream_t stream = 0);

// Embedding lookup
void gather_embeddings(
    const int* tokens,      // [num_tokens]
    const half* embedding,  // [vocab_size, hidden_dim]
    half* output,           // [num_tokens, hidden_dim]
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

Rust-like Result type for exception-free error propagation.

```cpp
#include <tiny_llm/result.h>

template<typename T>
class Result {
public:
    static Result<T> ok(T value);
    static Result<T> err(std::string message);
    
    bool isOk() const;
    bool isErr() const;
    
    T& value();              // throws if err
    const T& value() const;
    const std::string& error() const;  // throws if ok
    
    T valueOr(T default_value) const;
    
    // Monadic operations
    template<typename F>
    auto map(F&& f) -> Result<decltype(f(value()))>;
    
    template<typename F>
    auto flatMap(F&& f) -> decltype(f(value()));
};

// void specialization
template<>
class Result<void> {
public:
    static Result<void> ok();
    static Result<void> err(std::string message);
    bool isOk() const;
    bool isErr() const;
    const std::string& error() const;
};
```

### CudaException

CUDA error exception.

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

// CUDA error checking macro
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
    DeviceBuffer();
    explicit DeviceBuffer(size_t count);
    ~DeviceBuffer();
    
    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Movable
    DeviceBuffer(DeviceBuffer&&) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
    
    T* data();
    const T* data() const;
    size_t size() const;
    size_t bytes() const;
    
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

### StreamPool

CUDA stream pool for parallel execution.

```cpp
class StreamPool {
public:
    explicit StreamPool(int num_streams = 4);
    
    cudaStream_t getStream();         // Round-robin
    cudaStream_t getStream(int idx);   // Get by index
    void synchronizeAll();
    int numStreams() const;
};
```

### CudaEvent

CUDA event for timing.

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

---

## Memory Alignment

```cpp
#include <tiny_llm/cuda_streams.h>

constexpr size_t GPU_MEMORY_ALIGNMENT = 128;

inline size_t alignTo(size_t size, size_t alignment);
inline void* allocateAligned(size_t size);
```

---

**Languages**: [English](API) | [中文](../zh/API)

[← Home](../../) | [Changelog](../../changelog/) | [Contributing](../../CONTRIBUTING)
