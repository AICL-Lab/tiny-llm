---
layout: default
title: "API 参考 — Tiny-LLM"
description: "Tiny-LLM 推理引擎完整 API 参考"
nav_order: 4
---

# API 参考

Tiny-LLM 推理引擎的完整 API 参考文档。

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

模型配置结构体，定义所有超参数。

```cpp
#include <tiny_llm/inference_engine.h>

struct ModelConfig {
    int vocab_size = 32000;           // 词表大小
    int hidden_dim = 4096;            // 隐藏层维度
    int num_layers = 32;              // Transformer 层数
    int num_heads = 32;               // 注意力头数
    int num_kv_heads = 32;            // KV 头数 (支持 GQA)
    int head_dim = 128;               // 每头维度
    int intermediate_dim = 11008;     // FFN 中间层维度
    int max_seq_len = 2048;           // 最大序列长度
    float rope_theta = 10000.0f;      // RoPE 基频
    float rms_norm_eps = 1e-5f;       // RMSNorm epsilon
    int eos_token_id = 2;             // 结束符 token ID
    int bos_token_id = 1;             // 起始符 token ID
};
```

**常见配置**:

| 模型大小 | hidden_dim | num_layers | num_heads | intermediate_dim |
|------------|------------|------------|-----------|------------------|
| 7B | 4096 | 32 | 32 | 11008 |
| 13B | 5120 | 40 | 40 | 13824 |
| 70B | 8192 | 80 | 64 | 28672 |

### GenerationConfig

文本生成配置，控制采样行为。

```cpp
struct GenerationConfig {
    int max_new_tokens = 256;         // 最大生成 token 数
    float temperature = 1.0f;         // 采样温度
    int top_k = 50;                   // Top-k 采样截断
    float top_p = 0.9f;               // Top-p (核) 采样阈值
    bool do_sample = false;           // 启用采样 (false = 贪婪)
    float repetition_penalty = 1.0f;  // 重复 token 惩罚
};
```

**采样参数**:

| 参数 | 范围 | 效果 |
|-----------|-------|--------|
| `temperature` | 0.0 - 2.0 | 越低越确定 |
| `top_k` | 1 - vocab_size | 只考虑概率最高的 k 个 token |
| `top_p` | 0.0 - 1.0 | 从累积概率达到 p 的 token 中采样 |
| `repetition_penalty` | 1.0 - 2.0 | >1.0 惩罚重复 token |

### QuantizedWeight

INT8 量化权重，带 FP16 缩放因子。

```cpp
struct QuantizedWeight {
    int8_t* data;                     // INT8 权重 [rows, cols]
    half* scales;                     // FP16 缩放 [rows/group_size, cols]
    int rows;                         // 输入维度
    int cols;                         // 输出维度
    int group_size = 128;             // 量化组大小
    
    // 辅助方法
    int scaleRows() const;            // ceil(rows / group_size)
    int scaleCols() const;            // cols
    size_t weightElements() const;    // rows * cols
    size_t scaleElements() const;     // scaleRows() * cols
    size_t weightBytes() const;       // weightElements()
    size_t scaleBytes() const;        // scaleElements() * 2
    size_t totalBytes() const;        // weightBytes + scaleBytes
    bool isValid() const;             // 验证维度
};
```

### GenerationStats

文本生成的性能统计。

```cpp
struct GenerationStats {
    float prefill_time_ms = 0.0f;     // Prefill 阶段时间 (ms)
    float decode_time_ms = 0.0f;      // Decode 阶段时间 (ms)
    int prompt_tokens = 0;            // Prompt token 数
    int tokens_generated = 0;         // 生成 token 数
    float tokens_per_second = 0.0f;   // 生成吞吐
    size_t peak_memory_bytes = 0;     // 峰值 GPU 显存使用
};
```

---

## 核心类

### InferenceEngine

主推理引擎类。对不同引擎实例的并发生成是线程安全的。

```cpp
#include <tiny_llm/inference_engine.h>

class InferenceEngine {
public:
    // 从自定义二进制格式加载模型
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& model_path,
        const ModelConfig& config
    );
    
    // 为 prompt 生成补全
    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& gen_config
    );
    
    // 获取生成统计
    const GenerationStats& getStats() const;
    void resetStats();
    
    // 独立采样函数（无状态）
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

**使用示例**:

```cpp
// 配置模型
ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

// 加载模型
auto result = InferenceEngine::load("model.bin", config);
if (result.isErr()) {
    std::cerr << "Error: " << result.error() << std::endl;
    return 1;
}
auto engine = std::move(result.value());

// 配置生成
GenerationConfig gen_config;
gen_config.max_new_tokens = 256;
gen_config.temperature = 0.7f;
gen_config.top_p = 0.9f;
gen_config.do_sample = true;

// 生成
std::vector<int> prompt = {1, 15043, 29892};  // "Hello,"
auto output = engine->generate(prompt, gen_config);

// 查看性能
const auto& stats = engine->getStats();
std::cout << "速度: " << stats.tokens_per_second << " tok/s" << std::endl;
```

### KVCacheManager

用于自回归生成的高效键值缓存管理。

```cpp
#include <tiny_llm/kv_cache.h>

struct KVCacheConfig {
    int num_layers = 32;              // Transformer 层数
    int num_heads = 32;               // KV 头数
    int head_dim = 128;               // 每头维度
    int max_seq_len = 2048;           // 最大序列长度
    int max_batch_size = 1;           // 最大 batch 大小
};

class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& config);
    ~KVCacheManager();
    
    // 序列管理
    Result<int> allocateSequence(int max_len);
    void releaseSequence(int seq_id);
    bool hasSequence(int seq_id) const;
    
    // 获取用于注意力计算的缓存
    std::pair<half*, half*> getCache(int seq_id, int layer_idx);
    int getSeqLen(int seq_id) const;
    
    // 追加 KV（只写，无状态）
    void appendKV(int seq_id, int layer_idx,
                  const half* new_k, const half* new_v,
                  int num_tokens, cudaStream_t stream = 0);
    
    // 所有层完成后推进序列长度
    void advanceSeqLen(int seq_id, int num_tokens);
    
    // 显存统计
    size_t getUsedMemory() const;
    size_t getTotalMemory() const;
    size_t getFreeMemory() const;
    int getActiveSequenceCount() const;
};
```

**使用模式**:

```cpp
KVCacheConfig cache_config;
cache_config.num_layers = 32;
cache_config.num_heads = 32;
cache_config.head_dim = 128;
cache_config.max_seq_len = 2048;

KVCacheManager kv_cache(cache_config);

// 分配序列
auto seq_result = kv_cache.allocateSequence(1024);
if (seq_result.isErr()) {
    // 处理分配失败
}
int seq_id = seq_result.value();

// 前向传播各层
for (int i = 0; i < num_layers; i++) {
    layers[i]->forward(hidden_states, kv_cache, seq_id, position, stream);
}

// 所有层完成后推进序列长度
kv_cache.advanceSeqLen(seq_id, 1);

// 完成后释放
kv_cache.releaseSequence(seq_id);
```

### TransformerLayer

带注意力和 FFN 的单个 Transformer 层。

```cpp
#include <tiny_llm/transformer.h>

class TransformerLayer {
public:
    TransformerLayer(int layer_idx,
                     const TransformerWeights& weights,
                     const ModelConfig& config);
    
    // 单 token 前向 (decode 阶段)
    void forward(half* hidden_states,
                 KVCacheManager& kv_cache,
                 int seq_id,
                 int position,
                 cudaStream_t stream = 0);
    
    // 多 token 前向 (prefill 阶段)
    void forwardPrefill(half* hidden_states,
                        KVCacheManager& kv_cache,
                        int seq_id,
                        int seq_len,
                        cudaStream_t stream = 0);
    
    int getLayerIdx() const;
};
```

---

## CUDA Kernel

### W8A16 矩阵乘法

```cpp
#include <w8a16_matmul.cuh>

namespace tiny_llm::kernels {

// 主 W8A16 矩阵乘 kernel
void w8a16_matmul(
    const half* input,           // [M, K] FP16
    const int8_t* weight,        // [K, N] INT8 (列主序)
    const half* scales,          // [K/group_size, N] FP16
    half* output,                // [M, N] FP16
    int M,                       // Batch 大小
    int N,                       // 输出维度
    int K,                       // 输入维度
    int group_size,              // 量化组大小
    cudaStream_t stream = 0
);

// 权重反量化（用于测试/参考）
void dequantize_weights(
    const int8_t* weight_int8,
    const half* scales,
    half* weight_fp16,
    int K, int N,
    int group_size,
    cudaStream_t stream = 0
);

// 参考实现，用于验证
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

### Attention Kernel

```cpp
#include <attention.cuh>

namespace tiny_llm::kernels {

// Decode: 单 query token 对缓存的 KV
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

// Prefill: 多 token 带因果掩码
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

// 独立 softmax
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

// 原地 RMSNorm
void rmsnorm_inplace(
    half* x,                     // [batch, hidden_dim] (原地)
    const half* weight,
    int batch_size,
    int hidden_dim,
    float eps = 1e-5f,
    cudaStream_t stream = 0
);

}
```

### 逐元素操作

```cpp
#include <elementwise.cuh>

namespace tiny_llm::kernels {

// 原地加法: data[i] += add[i]
void add_inplace(
    half* data,
    const half* add,
    int num_elements,
    cudaStream_t stream = 0
);

// 融合 SwiGLU: gate[i] = silu(gate[i]) * up[i]
void silu_mul_inplace(
    half* gate,
    const half* up,
    int num_elements,
    cudaStream_t stream = 0
);

// Embedding 查找
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

## 错误处理

### Result<T>

受 Rust 启发的 Result 类型，无需异常的错误处理。

```cpp
#include <tiny_llm/result.h>

template<typename T>
class Result {
public:
    // 构造函数
    static Result<T> ok(T value);
    static Result<T> err(std::string message);
    
    // 状态检查
    bool isOk() const;
    bool isErr() const;
    
    // 获取值（error 时抛出）
    T& value();
    const T& value() const;
    T valueOr(T default_value) const;
    
    // 获取错误（ok 时抛出）
    const std::string& error() const;
    
    // Monadic 操作
    template<typename F>
    auto map(F&& f) -> Result<decltype(f(value()))>;
    
    template<typename F>
    auto flatMap(F&& f) -> decltype(f(value()));
};
```

**使用**:

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

// 或使用默认值
int val = parseInt("abc").valueOr(0);  // val = 0
```

### CudaException

带上下文的 CUDA 错误异常。

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

// 错误检查宏
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

GPU 显存的 RAII 封装。

```cpp
#include <tiny_llm/cuda_utils.h>

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer();                        // 空缓存
    explicit DeviceBuffer(size_t count);   // 分配 count 个元素
    ~DeviceBuffer();                       // 自动清理
    
    // 不可复制
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // 可移动
    DeviceBuffer(DeviceBuffer&&) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
    
    // 数据访问
    T* data();
    const T* data() const;
    size_t size() const;
    size_t bytes() const;
    
    // 数据传输
    void copyFromHost(const T* src, size_t count, cudaStream_t stream = 0);
    void copyToHost(T* dst, size_t count, cudaStream_t stream = 0) const;
};
```

### CudaStream

CUDA 流的 RAII 封装。

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

用于计时和同步的 CUDA 事件。

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

**计时示例**:

```cpp
CudaEvent start, end;

start.record(stream);
kernel<<<grid, block, 0, stream>>>(...);
end.record(stream);

end.synchronize();
float ms = CudaEvent::elapsedMs(start, end);
std::cout << "Kernel 时间: " << ms << " ms" << std::endl;
```

### StreamPool

用于并行执行的 CUDA 流池。

```cpp
class StreamPool {
public:
    explicit StreamPool(int num_streams = 4);
    
    cudaStream_t getStream();        // 轮询
    cudaStream_t getStream(int idx);  // 按索引
    
    void synchronizeAll();
    int numStreams() const;
};
```

---

**Languages**: [English](../en/API) | [中文](API) | [开发者指南 →](DEVELOPER)

[← 架构设计](ARCHITECTURE) | [开发者指南 →](DEVELOPER)
