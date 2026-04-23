---
layout: default
title: "架构设计 — Tiny-LLM"
description: "Tiny-LLM 系统架构和设计文档"
nav_order: 3
---

# 架构设计

Tiny-LLM 推理引擎的系统架构和设计文档。

---

## 目录

- [概述](#概述)
- [高层架构](#高层架构)
- [核心组件](#核心组件)
- [数据流](#数据流)
- [内存布局](#内存布局)
- [性能优化](#性能优化)
- [设计原则](#设计原则)

---

## 概述

Tiny-LLM 是一个高性能 CUDA C++ 推理引擎，专为高效的 Transformer 模型推理而设计。其核心特性包括：

| 特性 | 技术 | 收益 |
|---------|------------|---------|
| **W8A16 量化** | INT8 权重 + FP16 激活 | 显存减少约 50% |
| **高效 KV 缓存** | 增量解码与序列管理 | O(1) 自回归步进 |
| **优化 Kernel** | Tensor Core INT8、共享内存 tiling | 最大吞吐 |
| **模块化设计** | 清晰职责分离 | 易于扩展和测试 |

---

## 高层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        InferenceEngine                               │
├─────────────────────────────────────────────────────────────────────┤
│  模型加载器                                                │
│  ├── 自定义二进制格式解析                                  │
│  ├── 权重反量化 (INT8 → FP16)                              │
│  └── GPU 显存分配                                          │
├─────────────────────────────────────────────────────────────────────┤
│  Transformer 层 (× num_layers)                             │
│  │                                                         │
│  ├─→ 自注意力层                                            │
│  │   ├── RMSNorm                                           │
│  │   ├── QKV 投影 (W8A16 MatMul)                           │
│  │   ├── RoPE 位置编码                                     │
│  │   ├── 注意力计算 (Decode/Prefill)                       │
│  │   ├── 输出投影 (W8A16 MatMul)                           │
│  │   └── 残差连接                                          │
│  │                                                         │
│  └─→ FFN 层                                                │
│      ├── RMSNorm                                           │
│      ├── Gate 投影 + SiLU (W8A16)                          │
│      ├── Up 投影 (W8A16)                                   │
│      ├── Down 投影 (W8A16)                                 │
│      └── 残差连接                                          │
├─────────────────────────────────────────────────────────────────────┤
│  输出处理                                                  │
│  ├── 最终 RMSNorm                                          │
│  ├── LM Head 投影                                          │
│  └── Token 采样 (Greedy/Top-k/Top-p)                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. InferenceEngine

模型推理的主入口。

```cpp
class InferenceEngine {
public:
    // 从磁盘加载模型
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& path, const ModelConfig& config);
    
    // 完整的生成流程
    std::vector<int> generate(
        const std::vector<int>& prompt, 
        const GenerationConfig& config);
    
    // 统计和性能分析
    const GenerationStats& getStats() const;
    void resetStats();
};
```

**核心职责**:
- 模型生命周期管理
- Prefill/decode 编排
- Token 采样和生成循环
- 性能分析

### 2. KV Cache 管理器

用于自回归生成的高效键值缓存。

**设计动机**: v2.0 重新设计修复了一个关键问题，即层顺序影响写入位置。新设计使用显式的序列长度推进。

```cpp
class KVCacheManager {
public:
    // 分配新的序列槽位
    Result<int> allocateSequence(int max_len);
    
    // 为特定层追加 KV (无状态)
    void appendKV(int seq_id, int layer_idx,
                  const half* k, const half* v,
                  int num_tokens, cudaStream_t stream);
    
    // 所有层完成后推进序列长度
    void advanceSeqLen(int seq_id, int num_tokens);
    
    // 访问缓存的 K/V 用于注意力计算
    std::pair<half*, half*> getCache(int seq_id, int layer_idx);
};
```

**内存布局**:
```
K 缓存: [max_batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
V 缓存: [max_batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
```

### 3. W8A16 量化

仅权重的 INT8 量化，使用 FP16 激活。

**量化方案**:
```
权重: INT8 [rows, cols]
缩放: FP16 [ceil(rows/group_size), cols]
输出: FP16 = dequantize(权重, 缩放) @ 激活_FP16
```

**组级量化**: 权重沿输入维度分为每组 128 个元素的组，每组共享一个缩放因子。

```
┌────────────────────────────────────────────────┐
│  QuantizedWeight 布局                          │
├────────────────────────────────────────────────┤
│  weights: int8_t [rows, cols]                  │
│  ├── group 0 (128 元素) → scales[0, :]         │
│  ├── group 1 (128 元素) → scales[1, :]         │
│  └── ...                                       │
│  scales: half [rows/128, cols]                 │
└────────────────────────────────────────────────┘
```

**优势**:
- 权重显存减少 50%
- 不量化激活 (保持精度)
- Ampere+ 上高效的 INT8 Tensor Core 利用

### 4. CUDA Kernel 实现

#### W8A16 矩阵乘法

```cpp
void w8a16_matmul(
    const half* input,      // [M, K] FP16
    const int8_t* weight,   // [K, N] INT8
    const half* scales,     // [K/128, N] FP16
    half* output,           // [M, N] FP16
    int M, int N, int K,
    int group_size = 128,
    cudaStream_t stream = 0);
```

**优化**:
- A 矩阵共享内存 tiling
- 向量化加载 (4 字节对齐)
- Warp shuffle 归约
- 合并写回

#### Attention Kernel

**Decode Attention** (单 token vs 缓存 KV):
```cpp
void attention_decode(
    const half* query,      // [batch, heads, 1, head_dim]
    const half* k_cache,    // [batch, heads, seq_len, head_dim]
    const half* v_cache,    // [batch, heads, seq_len, head_dim]
    half* output,
    float scale,            // 1/sqrt(head_dim)
    int batch_size, int heads, int seq_len, int head_dim,
    cudaStream_t stream);
```

**Prefill Attention** (多 token 带因果掩码):
```cpp
void attention_prefill(
    const half* query,      // [batch, heads, seq_len, head_dim]
    const half* key,
    const half* value,
    half* output,
    float scale,
    int batch_size, int heads, int seq_len, int head_dim,
    cudaStream_t stream);
```

**优化**:
- 在线 softmax 数值稳定性
- 缓存访问内存合并
- Kernel 融合机会

---

## 数据流

### Prefill 阶段 (Prompt 处理)

```
输入 Tokens (B, S)
      │
      ▼
┌─────────────┐
│ Embedding   │ (B, S, H)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Transformer 层 (× N)                    │
│                                         │
│  ┌─────────┐    ┌─────────┐            │
│  │ RMSNorm │───▶│ QKV W8A │            │
│  └────┬────┘    │  MatMul │            │
│       │         └────┬────┘            │
│       │              │                 │
│       │              ▼                 │
│       │         ┌─────────┐            │
│       │         │  RoPE   │            │
│       │         └────┬────┘            │
│       │              │                 │
│       │              ▼                 │
│       │    ┌──────────────────┐        │
│       │    │ Attention Prefill│        │
│       │    │ (causal mask)    │        │
│       │    └────────┬─────────┘        │
│       │             │                  │
│       │    ┌────────┴────────┐         │
│       │    │  KV Cache Write │ (all pos)│
│       │    └─────────────────┘         │
│       │             │                  │
│       │             ▼                  │
│  ┌────┴─────┐  ┌─────────┐             │
│  │ Residual │◄─│ Out Proj│             │
│  └────┬─────┘  │ W8A16   │             │
│       │        └─────────┘             │
│       │           ...                  │
└───────┼───────────────────────────────┘
        │
        ▼
┌─────────────┐
│ Final Norm  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ LM Head     │ ──▶ Logits (B, S, V)
└─────────────┘
```

### Decode 阶段 (Token 生成)

```
单 Token (B, 1)
      │
      ▼
┌─────────────┐
│ Embedding   │ (B, 1, H)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Transformer 层 (× N)                    │
│                                         │
│  ┌─────────┐    ┌─────────┐            │
│  │ RMSNorm │───▶│ QKV W8A │            │
│  └────┬────┘    │  MatMul │            │
│       │         └────┬────┘            │
│       │              │                 │
│       │              ▼                 │
│       │         ┌─────────┐            │
│       │         │  RoPE   │            │
│       │         └────┬────┘            │
│       │              │                 │
│       │              ▼                 │
│       │    ┌──────────────────┐        │
│       │    │ Attention Decode │        │
│       │    │ (single token)   │        │
│       │    │  • Read KV Cache │        │
│       │    │  • Compute attn  │        │
│       │    └────────┬─────────┘        │
│       │             │                  │
│       │    ┌────────┴────────┐         │
│       │    │  KV Cache Append│ (new KV) │
│       │    └─────────────────┘         │
│       │             │                  │
│       │             ▼                  │
│  ┌────┴─────┐  ┌─────────┐             │
│  │ Residual │◄─│ Out Proj│             │
│  └────┬─────┘  │ W8A16   │             │
│       │        └─────────┘             │
│       │           ...                  │
└───────┼───────────────────────────────┘
        │
        ▼
┌─────────────┐
│ Final Norm  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ LM Head     │
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌─────────┐
│  Logits     │───▶│ Sampling│── Next Token
│  (B, 1, V)  │    │ (temp/k/p)│
└─────────────┘    └─────────┘
```

---

## 内存布局

### 权重存储

```
┌─────────────────────────────────────────────────────────────────────┐
│                         模型权重                                     │
├─────────────────────────────────────────────────────────────────────┤
│ Token 词嵌入                                                         │
│ [vocab_size, hidden_dim] FP16                                        │
│ ~250 MB (32k vocab, 4k hidden)                                       │
├─────────────────────────────────────────────────────────────────────┤
│ 层权重 (× num_layers)                                                │
│                                                                      │
│  [注意力]                                                            │
│  ├── q_proj:  INT8 [hidden_dim, hidden_dim]                          │
│  ├── k_proj:  INT8 [hidden_dim, num_kv_heads × head_dim]              │
│  ├── v_proj:  INT8 [hidden_dim, num_kv_heads × head_dim]              │
│  ├── o_proj:  INT8 [hidden_dim, hidden_dim]                          │
│  └── 缩放: FP16 (每层不同大小)                                       │
│                                                                      │
│  [FFN - SwiGLU]                                                      │
│  ├── gate_proj: INT8 [hidden_dim, intermediate_dim]                  │
│  ├── up_proj:   INT8 [hidden_dim, intermediate_dim]                  │
│  ├── down_proj: INT8 [intermediate_dim, hidden_dim]                  │
│  └── 缩放: FP16 (每层不同大小)                                       │
│                                                                      │
│  每层: ~500 MB (7B 模型, 4k/11k 维度, W8A16)                         │
│  总计 (32 层): ~16 GB                                                │
├─────────────────────────────────────────────────────────────────────┤
│ 输出头                                                               │
│ [hidden_dim, vocab_size] FP16 (通常与嵌入层共享)                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 激活缓存

| 缓存 | 形状 | 数据类型 | 大小 (B=1, S=2048, H=4096) |
|--------|-------|-------|---------------------------|
| 隐藏状态 | [B, S, H] | FP16 | 16 MB |
| 注意力输出 | [B, heads, S, head_dim] | FP16 | 16 MB |
| QKV | [B, S, 3×H] | FP16 | 48 MB |
| FFN 中间结果 | [B, S, intermediate_dim] | FP16 | 44 MB |

---

## 性能优化

### 内存优化

| 技术 | 实现 | 收益 |
|-----------|----------------|---------|
| W8A16 量化 | 每组 INT8 权重 + FP16 缩放 | 权重显存减少 50% |
| KV Cache 分页 | 预分配 + 序列管理 | 高效批处理 |
| 激活复用 | 原地操作 | 减少分配 |

### 计算优化

| 技术 | 应用 | 加速比 |
|-----------|-------------|---------|
| Tensor Cores | INT8 矩阵乘 (Ampere+) | 2-4× vs FP16 |
| Warp Shuffle | 归约 | 消除共享内存 |
| 向量化加载 | 128-bit 内存访问 | 更好带宽 |
| Kernel 融合 | RMSNorm+Resid, SiLU+Mul | 减少内核启动 |

### 优化的 Kernel 列表

| Kernel | 优化 | 吞吐 |
|--------|---------------|------------|
| `w8a16_matmul` | Tiling, 向量化, warp shuffle | ~80% Tensor Core |
| `attention_decode` | 在线 softmax, 融合 KV 读取 | 显存带宽瓶颈 |
| `attention_prefill` | Tiled softmax, 融合因果掩码 | 计算瓶颈 |
| `rmsnorm` | Warp 归约, 向量化 | >1TB/s 带宽 |
| `rope` | 缓存三角函数, 向量化 | 可忽略开销 |

---

## 设计原则

1. **模块化**: 层、Kernel 和工具之间清晰的接口
2. **类型安全**: Result<T> 错误处理，全程强类型
3. **RAII**: GPU 显存和流的自动资源管理
4. **可测试性**: 全面的单元测试和基于属性的测试
5. **可扩展性**: 易于添加新的 Kernel、采样策略、模型格式

---

**Languages**: [English](../en/ARCHITECTURE) | [中文](ARCHITECTURE) | [API 参考 →](API)

[← 快速开始](QUICKSTART) | [开发者指南 →](DEVELOPER)
