# Tiny-LLM API 文档

## 数据类型

### ModelConfig

模型配置结构体。

```cpp
struct ModelConfig {
    int vocab_size = 32000;      // 词表大小
    int hidden_dim = 4096;       // 隐藏层维度
    int num_layers = 32;         // Transformer 层数
    int num_heads = 32;          // 注意力头数
    int num_kv_heads = 32;       // KV 头数 (GQA)
    int head_dim = 128;          // 每个头的维度
    int intermediate_dim = 11008; // FFN 中间层维度
    int max_seq_len = 2048;      // 最大序列长度
    float rope_theta = 10000.0f; // RoPE 参数
    float rms_norm_eps = 1e-5f;  // RMSNorm epsilon
    int eos_token_id = 2;        // EOS token ID
    int bos_token_id = 1;        // BOS token ID
};
```

### GenerationConfig

生成配置结构体。

```cpp
struct GenerationConfig {
    int max_new_tokens = 256;    // 最大生成 token 数
    float temperature = 1.0f;    // 温度参数
    int top_k = 50;              // Top-k 采样参数
    float top_p = 0.9f;          // Top-p 采样参数
    bool do_sample = false;      // 是否采样 (false=贪婪)
    float repetition_penalty = 1.0f; // 重复惩罚
};
```

### QuantizedWeight

量化权重结构体。

```cpp
struct QuantizedWeight {
    int8_t* data;     // INT8 量化权重 [rows, cols]
    half* scales;     // Scale 因子 [rows, cols/group_size]
    int rows;         // 行数
    int cols;         // 列数
    int group_size;   // 量化组大小
    
    bool isValid() const;           // 验证有效性
    size_t weightBytes() const;     // 权重字节数
    size_t scaleBytes() const;      // Scale 字节数
    size_t totalBytes() const;      // 总字节数
};
```

## 核心类

### InferenceEngine

推理引擎主类。

```cpp
class InferenceEngine {
public:
    // 从文件加载模型
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& model_path,
        const ModelConfig& config
    );
    
    // 生成 token
    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& config
    );
    
    // 获取统计信息
    const GenerationStats& getStats() const;
    
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

### KVCacheManager

KV Cache 管理器。

```cpp
class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& config);
    
    // 分配新序列的 cache
    Result<int> allocateSequence(int max_len);
    
    // 释放序列的 cache
    void releaseSequence(int seq_id);
    
    // 获取 cache 指针
    std::pair<half*, half*> getCache(int seq_id, int layer_idx);
    
    // 追加 KV 对
    void appendKV(int seq_id, int layer_idx, 
                  const half* new_k, const half* new_v, int num_tokens);
    
    // 获取序列长度
    int getSeqLen(int seq_id) const;
    
    // 内存统计
    size_t getUsedMemory() const;
    size_t getTotalMemory() const;
};
```

### TransformerLayer

Transformer 层。

```cpp
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
};
```

## CUDA Kernel

### W8A16 矩阵乘法

```cpp
namespace tiny_llm::kernels {

// W8A16 量化矩阵乘法
void w8a16_matmul(
    const half* input,      // [M, K] FP16 输入
    const int8_t* weight,   // [K, N] INT8 权重
    const half* scales,     // [K/group_size, N] scale 因子
    half* output,           // [M, N] FP16 输出
    int M, int N, int K,
    int group_size,
    cudaStream_t stream = 0
);

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

### 注意力计算

```cpp
namespace tiny_llm::kernels {

// Decode 注意力 (单 token)
void attention_decode(
    const half* query,      // [batch, num_heads, 1, head_dim]
    const half* k_cache,    // [batch, num_heads, seq_len, head_dim]
    const half* v_cache,    // [batch, num_heads, seq_len, head_dim]
    half* output,           // [batch, num_heads, 1, head_dim]
    float scale,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

// Prefill 注意力 (多 token，带因果掩码)
void attention_prefill(
    const half* query,
    const half* key,
    const half* value,
    half* output,
    float scale,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = 0
);

}
```

### RMSNorm

```cpp
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

}
```

## 错误处理

### Result<T>

类似 Rust 的 Result 类型。

```cpp
template<typename T>
class Result {
public:
    static Result<T> ok(T value);
    static Result<T> err(std::string message);
    
    bool isOk() const;
    bool isErr() const;
    
    T& value();
    const std::string& error() const;
};

// 使用示例
auto result = InferenceEngine::load("model.bin", config);
if (result.isErr()) {
    std::cerr << "Error: " << result.error() << std::endl;
    return 1;
}
auto engine = std::move(result.value());
```

### CUDA_CHECK

CUDA 错误检查宏。

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw CudaException(err, __FILE__, __LINE__); \
        } \
    } while(0)

// 使用示例
CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
```

## 工具类

### DeviceBuffer<T>

GPU 内存缓冲区 RAII 封装。

```cpp
template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count);
    ~DeviceBuffer();
    
    T* data();
    size_t size() const;
    
    void copyFromHost(const T* src, size_t count);
    void copyToHost(T* dst, size_t count);
};
```

### StreamPool

CUDA Stream 池。

```cpp
class StreamPool {
public:
    explicit StreamPool(int num_streams = 4);
    
    cudaStream_t getStream();           // 轮询获取
    cudaStream_t getStream(int idx);    // 指定索引
    void synchronizeAll();              // 同步所有 stream
    int numStreams() const;
};
```
