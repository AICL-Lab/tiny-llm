# Tiny-LLM 设计文档

> 版本: 2.0.1 | 日期: 2026-04-16

## 概述

轻量级 LLM 推理引擎，专注于 W8A16 量化推理与 CUDA Kernel 实现。

### 设计目标

| 目标 | 描述 |
|------|------|
| 显存效率 | INT8 权重量化，减少 ~50% 显存 |
| 带宽优化 | 寄存器级反量化，最大化带宽利用 |
| 计算效率 | Warp shuffle + shared memory tiling |
| 可扩展性 | 模块化设计，支持不同模型架构 |

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Host (CPU)                             │
│  Model Loader  │  Tokenizer  │  Generation Controller      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Device (GPU)                           │
│                                                             │
│  ┌─ Memory ─────────────────────────────────────────────┐  │
│  │  Weight Memory (INT8+Scales)  │  KV Cache  │  Activ  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─ Kernels ────────────────────────────────────────────┐  │
│  │  W8A16 MatMul  │  Attention  │  RMSNorm  │  Elemwise │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─ Transformer ────────────────────────────────────────┐  │
│  │  Layer 1  →  Layer 2  →  ...  →  Layer N            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### Model Loader

```cpp
struct ModelConfig {
    int vocab_size, hidden_dim, num_layers;
    int num_heads, num_kv_heads, head_dim;
    int intermediate_dim, max_seq_len;
    float rope_theta, rms_norm_eps;
};

struct QuantizedWeight {
    int8_t* data;      // INT8 权重
    half* scales;      // Per-group scales
    int rows, cols, group_size;
};
```

### W8A16 MatMul

计算公式: `output = input @ (weight_int8 * scales)`

优化技术:
- Shared memory tiling: TILE_M × TILE_N × BLOCK_K
- Warp shuffle reduction
- Register-level dequantization

### KV Cache Manager

关键 API:
```cpp
Result<int> allocateSequence(int max_len);
void appendKV(seq_id, layer_idx, k, v, n);  // 只写入
void advanceSeqLen(seq_id, n);              // 显式推进
void releaseSequence(int seq_id);
```

---

## 正确性属性

| ID | 属性 | 验证需求 |
|----|------|----------|
| P1 | W8A16 相对误差 < 1% | 2.5, 2.6 |
| P2 | KV Cache 不变量 | 3.2-3.6 |
| P3 | 因果掩码正确 | 4.2 |
| P4 | RMSNorm 输出 RMS ≈ 1 | 4.4 |
| P5 | 增量解码等价全量计算 | 4.6 |
| P6 | 贪婪采样 = argmax | 5.2 |
| P7 | 生成长度 ≤ max_new_tokens | 5.4 |
| P8 | Scale 维度正确 | 1.3, 7.2 |
| P9 | 损坏文件不崩溃 | 1.5 |

---

## 错误处理

```cpp
// CUDA 错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) throw CudaException(err, __FILE__, __LINE__); \
    } while(0)

// Result 类型
template<typename T>
class Result {
    static Result<T> ok(T value);
    static Result<T> err(std::string message);
    bool isOk() const;
    T& value();
    const std::string& error() const;
};
```

---

## 测试策略

| 类型 | 框架 | 覆盖 |
|------|------|------|
| 单元测试 | GoogleTest | 具体示例、边界情况 |
| 属性测试 | RapidCheck | 通用正确性 (100+ 次) |
| 集成测试 | GoogleTest | 端到端推理 |

---

## 文件结构

```
tiny-llm/
├── include/tiny_llm/     # 公共头文件
│   ├── types.h           # 核心类型
│   ├── result.h          # 错误处理
│   ├── cuda_utils.h      # CUDA 工具
│   ├── kv_cache.h        # KV Cache
│   ├── transformer.h     # Transformer 层
│   ├── model_loader.h    # 模型加载
│   └── inference_engine.h
├── kernels/              # CUDA Kernel
│   ├── w8a16_matmul.cu
│   ├── attention.cu
│   ├── rmsnorm.cu
│   └── elementwise.cu
├── src/                  # 实现
└── tests/                # 测试
```
