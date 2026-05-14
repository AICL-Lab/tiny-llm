# 配置

本指南介绍 Tiny-LLM 的所有配置选项。

## 模型配置

`ModelConfig` 结构体定义模型超参数：

```cpp
struct ModelConfig {
    size_t vocab_size;       // 词汇表大小
    size_t hidden_dim;       // 隐藏层维度
    size_t intermediate_dim; // FFN 中间维度
    size_t num_layers;       // Transformer 层数
    size_t num_heads;        // 注意力头数
    size_t num_kv_heads;     // KV 头数（用于 GQA）
    size_t head_dim;         // 头维度
    size_t max_seq_len;      // 最大序列长度
    float rope_theta;        // RoPE theta 参数
    float rms_norm_eps;      // RMS 归一化 epsilon
};
```

### 示例配置

```cpp
// LLaMA-7B 风格配置
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

## 生成配置

`GenerationConfig` 结构体控制文本生成：

```cpp
struct GenerationConfig {
    size_t max_tokens;      // 最大生成 token 数
    float temperature;      // 采样温度
    float top_p;            // 核采样阈值
    size_t top_k;           // Top-k 采样
    float repetition_penalty; // 重复惩罚
    size_t seed;            // 随机种子（0 = 随机）
};
```

### 采样策略

#### 贪心解码

```cpp
GenerationConfig config;
config.temperature = 0.0f;  // 贪心
```

#### 温度采样

```cpp
GenerationConfig config;
config.temperature = 0.7f;
```

#### Top-K 采样

```cpp
GenerationConfig config;
config.temperature = 0.9f;
config.top_k = 50;
```

#### 核采样 (Top-P)

```cpp
GenerationConfig config;
config.temperature = 0.9f;
config.top_p = 0.95f;
```

## KV 缓存配置

```cpp
struct KVCacheConfig {
    size_t max_batch_size;  // 最大并发序列数
    size_t max_seq_len;     // 最大序列长度
    bool enable_swapping;   // 启用 CPU 交换
};
```

## 量化配置

```cpp
struct QuantizationConfig {
    QuantizationType type;  // INT8, INT4 等
    size_t group_size;      // 每组量化大小
    bool symmetrical;       // 对称 vs 非对称
};
```

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `TINYLLM_CUDA_DEVICE` | CUDA 设备 ID | `0` |
| `TINYLLM_CACHE_DIR` | 模型缓存目录 | `~/.tinyllm/cache` |
| `TINYLLM_LOG_LEVEL` | 日志级别 | `INFO` |

## 下一步

- [API 参考](/zh/api/) - 完整的 API 文档
- [性能指南](/zh/performance/optimization) - 优化技术
