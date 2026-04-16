---
layout: default
title: "API 参考 — Tiny-LLM"
description: "Tiny-LLM 推理引擎公共接口完整文档"
nav_order: 3
---

# API 参考

Tiny-LLM 推理引擎的完整公共接口文档。

---

## 目录

- [数据类型](#数据类型)
- [核心类](#核心类)
- [CUDA Kernel](#cuda-kernel)
- [错误处理](#错误处理)
- [工具类](#工具类)

---

## 数据类型

### ModelConfig

模型配置结构体，定义模型的核心参数。

```cpp
struct ModelConfig {
    int vocab_size = 32000;       // 词表大小
    int hidden_dim = 4096;        // 隐藏层维度
    int num_layers = 32;          // Transformer 层数
    int num_heads = 32;           // 注意力头数
    int num_kv_heads = 32;        // KV 头数 (GQA)
    int head_dim = 128;           // 每头维度
    int intermediate_dim = 11008; // FFN 中间层维度
    int max_seq_len = 2048;       // 最大序列长度
    float rope_theta = 10000.0f;  // RoPE 参数
    float rms_norm_eps = 1e-5f;   // RMSNorm epsilon
    int eos_token_id = 2;         // EOS token ID
    int bos_token_id = 1;         // BOS token ID
};
```

**参数说明**:

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `vocab_size` | 词表大小 | 32000 (LLaMA), 100277 (Qwen) |
| `hidden_dim` | 隐藏层维度 | 4096 (7B), 5120 (13B) |
| `num_layers` | Transformer 层数 | 32 (7B), 40 (13B) |
| `num_heads` | 注意力头数 | 32 (7B) |
| `num_kv_heads` | KV 头数 (GQA) | 32 (MHA), 8 (GQA) |
| `head_dim` | 每头维度 | 128 |
| `intermediate_dim` | FFN 中间层 | 11008 (7B) |
| `max_seq_len` | 最大序列长度 | 2048, 4096, 8192 |
| `rope_theta` | RoPE 基频 | 10000.0, 1000000.0 |

### GenerationConfig

生成配置结构体，控制文本生成行为。

```cpp
struct GenerationConfig {
    int max_new_tokens = 256;     // 最大生成 token 数
    float temperature = 1.0f;     // 温度参数
    int top_k = 50;               // Top-k 参数
    float top_p = 0.9f;           // Top-p 参数
    bool do_sample = false;       // 是否采样 (false=贪婪)
    float repetition_penalty = 1.0f;  // 重复惩罚系数
};
```

**采样参数说明**:

| 参数 | 范围 | 说明 |
|------|------|------|
| `temperature` | 0.0 - 2.0 | 越低越确定，越高越随机 |
| `top_k` | 1 - vocab_size | 只从概率最高的 k 个 token 采样 |
| `top_p` | 0.0 - 1.0 | 核采样，从累积概率达到 p 的 token 中采样 |
| `repetition_penalty` | 1.0 - 2.0 | >1.0 惩罚重复 token |

### QuantizedWeight

量化权重结构体，INT8 权重 + FP16 缩放因子。

```cpp
struct QuantizedWeight {
    int8_t* data;     // INT8 量化权重 [rows, cols]
    half* scales;     // Scale 因子 [ceil(rows/group_size), cols]
    int rows;
    int cols;
    int group_size;   // 量化组大小 (默认 128)
    
    // 方法
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

**内存布局**:

```
权重: INT8 [rows, cols]
      ┌─────────────────┐
      │  group 0 (128)  │ ──► scale[0, :]
      ├─────────────────┤
      │  group 1 (128)  │ ──► scale[1, :]
      ├─────────────────┤
      │      ...        │
      └─────────────────┘
      
缩放因子: FP16 [ceil(rows/128), cols]
```

### GenerationStats

生成统计结构体，记录性能指标。

```cpp
struct GenerationStats {
    float prefill_time_ms = 0.0f;    // Prefill 耗时
    float decode_time_ms = 0.0f;     // Decode 耗时
    int prompt_tokens = 0;            // Prompt token 数
    int tokens_generated = 0;         // 生成 token 数
    float tokens_per_second = 0.0f;  // 生成速度
    size_t peak_memory_bytes = 0;    // 峰值显存
};
```

---

## 核心类

### InferenceEngine

推理引擎主类，提供完整的推理能力。

```cpp
#include <tiny_llm/inference_engine.h>

class InferenceEngine {
public:
    // 从文件加载模型
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& model_path,
        const ModelConfig& config
    );
    
    // 生成 tokens
    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& config
    );
    
    // 获取统计信息
    const GenerationStats& getStats() const;
    void resetStats();
    
    // 采样函数 (静态，可单独使用)
    static int sampleGreedy(const half* logits, int vocab_size);
    static int sampleTemperature(const half* logits, int vocab_size, 
                                  float temperature, unsigned seed = 0);
    static int sampleTopK(const half* logits, int vocab_size, 
                          int k, float temperature, unsigned seed = 0);
    static int sampleTopP(const half* logits, int vocab_size, 
                          float p, float temperature, unsigned seed = 0);
};
```

**完整使用示例**:

```cpp
#include <tiny_llm/inference_engine.h>
#include <iostream>

int main() {
    // 1. 配置模型参数
    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;
    config.num_heads = 32;
    config.max_seq_len = 2048;
    
    // 2. 加载模型
    auto result = InferenceEngine::load("model.bin", config);
    if (result.isErr()) {
        std::cerr << "Error: " << result.error() << std::endl;
        return 1;
    }
    auto engine = std::move(result.value());
    
    // 3. 配置生成参数
    GenerationConfig gen_config;
    gen_config.max_new_tokens = 100;
    gen_config.temperature = 0.7f;
    gen_config.top_p = 0.9f;
    gen_config.do_sample = true;
    
    // 4. 生成文本
    std::vector<int> prompt = {1, 15043, 29892};  // "Hello," tokens
    auto output = engine->generate(prompt, gen_config);
    
    // 5. 查看性能统计
    const auto& stats = engine->getStats();
    std::cout << "生成速度: " << stats.tokens_per_second << " tokens/s" << std::endl;
    std::cout << "显存峰值: " << stats.peak_memory_bytes / 1024 / 1024 << " MB" << std::endl;
    
    return 0;
}
```

### KVCacheManager

KV Cache 管理器，管理增量解码的 key-value 缓存。

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
    
    // 序列管理
    Result<int> allocateSequence(int max_len);
    void releaseSequence(int seq_id);
    bool hasSequence(int seq_id) const;
    
    // Cache 访问
    std::pair<half*, half*> getCache(int seq_id, int layer_idx);
    
    // 追加 KV (只写入，不推进可见长度)
    void appendKV(int seq_id, int layer_idx,
                  const half* new_k, const half* new_v, 
                  int num_tokens, cudaStream_t stream = 0);
    
    // 显式推进序列长度 (所有层 append 后调用)
    void advanceSeqLen(int seq_id, int num_tokens);
    
    // 查询
    int getSeqLen(int seq_id) const;
    
    // 内存统计
    size_t getUsedMemory() const;
    size_t getTotalMemory() const;
    size_t getFreeMemory() const;
    int getActiveSequenceCount() const;
};
```

**关键设计模式**:

```cpp
// 正确的使用模式
for (auto &layer : layers_) {
    layer->forward(hidden_states, *kv_cache_, seq_id, position, stream_);
}
kv_cache_->advanceSeqLen(seq_id, 1);  // 显式更新长度
```

### TransformerLayer

Transformer 层实现。

```cpp
#include <tiny_llm/transformer.h>

class TransformerLayer {
public:
    TransformerLayer(int layer_idx, const TransformerWeights& weights, 
                     const ModelConfig& config);
    
    // 单 token 前向传播 (decode)
    void forward(half* hidden_states, KVCacheManager& kv_cache,
                 int seq_id, int position, cudaStream_t stream = 0);
    
    // 多 token 前向传播 (prefill)
    void forwardPrefill(half* hidden_states, KVCacheManager& kv_cache,
                        int seq_id, int seq_len, cudaStream_t stream = 0);
    
    int getLayerIdx() const;
};
```

---

## CUDA Kernel

### W8A16 矩阵乘法

```cpp
#include <w8a16_matmul.cuh>

namespace tiny_llm::kernels {

// W8A16 量化矩阵乘法
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

// 参考实现 (用于测试)
void w8a16_matmul_reference(...);

// 权重反量化
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

// Decode 注意力 (单 token vs cached K/V)
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

// Prefill 注意力 (多 token，带因果掩码)
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

### Elementwise

```cpp
#include <elementwise.cuh>

namespace tiny_llm::kernels {

// 原地加法: data[i] += add[i]
void add_inplace(half* data, const half* add, int num_elements,
                 cudaStream_t stream = 0);

// SiLU × multiply: gate[i] = silu(gate[i]) * up[i]
void silu_mul_inplace(half* gate, const half* up, int num_elements,
                      cudaStream_t stream = 0);

// Embedding 查找
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

## 错误处理

### Result<T>

类似 Rust 的 Result 类型，用于无异常错误传播。

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

// void 特化
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

**使用示例**:

```cpp
Result<int> parseInt(const std::string& s) {
    try {
        return Result<int>::ok(std::stoi(s));
    } catch (...) {
        return Result<int>::err("Invalid integer");
    }
}

auto result = parseInt("42");
if (result.isOk()) {
    std::cout << "Value: " << result.value() << std::endl;
} else {
    std::cerr << "Error: " << result.error() << std::endl;
}

// 或使用默认值
int val = parseInt("abc").valueOr(0);  // val = 0
```

### CudaException

CUDA 错误异常。

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

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw CudaException(err, __FILE__, __LINE__); \
        } \
    } while(0)
```

---

## 工具类

### DeviceBuffer<T>

GPU 内存 RAII 封装。

```cpp
#include <tiny_llm/cuda_utils.h>

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer();
    explicit DeviceBuffer(size_t count);
    ~DeviceBuffer();
    
    // 不可复制
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // 可移动
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

**使用示例**:

```cpp
// 分配设备内存
DeviceBuffer<float> d_buffer(1024);

// 从主机复制数据
std::vector<float> h_data(1024, 1.0f);
d_buffer.copyFromHost(h_data.data(), h_data.size());

// 使用数据
kernel<<<grid, block>>>(d_buffer.data());
```

### CudaStream

CUDA Stream RAII 封装。

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

CUDA Stream 池。

```cpp
class StreamPool {
public:
    explicit StreamPool(int num_streams = 4);
    
    cudaStream_t getStream();         // 轮询获取
    cudaStream_t getStream(int idx);   // 指定索引
    void synchronizeAll();
    int numStreams() const;
};
```

### CudaEvent

CUDA 事件计时。

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

**使用示例**:

```cpp
CudaEvent start, end;

start.record(stream);
kernel<<<grid, block, 0, stream>>>(...);
end.record(stream);

end.synchronize();
float elapsed = CudaEvent::elapsedMs(start, end);
std::cout << "Kernel time: " << elapsed << " ms" << std::endl;
```

---

## 内存对齐

```cpp
#include <tiny_llm/cuda_streams.h>

constexpr size_t GPU_MEMORY_ALIGNMENT = 128;

inline size_t alignTo(size_t size, size_t alignment);
inline void* allocateAligned(size_t size);
```

---

**Languages**: [English](../en/API) | [中文](API)

[← 返回首页](../../) | [更新日志](../../changelog/) | [贡献指南](../../CONTRIBUTING)
