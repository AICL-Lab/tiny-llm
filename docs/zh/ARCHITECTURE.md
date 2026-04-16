---
layout: default
title: "架构设计 — Tiny-LLM"
description: "Tiny-LLM 系统架构与设计原理"
nav_order: 2
---

# 架构设计

本文档描述 Tiny-LLM 的系统架构和设计原理。

---

## 概述

Tiny-LLM 是一个轻量级 CUDA/C++ 推理引擎，专注于：
- **W8A16 量化**：INT8 权重 + FP16 激活，显存减少约 50%
- **高效 KV Cache**：增量解码与序列管理
- **高性能 Kernel**：共享内存 tiling、warp shuffle 优化

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    InferenceEngine                           │
├─────────────────────────────────────────────────────────────┤
│  模型加载器 ──► 权重 (INT8 + FP16 缩放因子)                  │
├─────────────────────────────────────────────────────────────┤
│  Transformer 层 × N                                         │
│  ├── 注意力: W8A16 矩阵乘 + KV 缓存 + RoPE + 因果掩码       │
│  │   ├── Q, K, V 投影 (W8A16)                               │
│  │   ├── RoPE 位置编码                                      │
│  │   ├── 注意力计算 (decode/prefill)                        │
│  │   └── 输出投影 (W8A16)                                   │
│  └── FFN: W8A16 矩阵乘 + SwiGLU                             │
│      ├── Gate 投影 + SiLU                                   │
│      └── Up 投影逐元素乘                                    │
├─────────────────────────────────────────────────────────────┤
│  RMSNorm + 残差连接                                         │
├─────────────────────────────────────────────────────────────┤
│  采样: 贪婪 / 温度 / Top-k / Top-p                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. 推理引擎 (InferenceEngine)

推理引擎是主要的入口点，提供以下能力：

| 能力 | 说明 |
|------|------|
| 模型加载 | 支持自定义二进制格式 (计划支持 GGUF) |
| Token 生成 | 提供 Prefill 和 Decode 两个阶段 |
| 采样策略 | 贪婪、温度、Top-k、Top-p |
| 性能统计 | 性能指标和显存使用追踪 |

### 2. KV 缓存管理器 (KVCacheManager)

用于增量解码的高效键值缓存：

| 特性 | 说明 |
|---------|-------------|
| 预分配内存 | 所有缓存内存预先分配 |
| 序列管理 | 动态分配/释放序列 |
| 无状态追加 | 写入位置与层顺序无关 |
| 显式推进 | 调用者控制序列长度推进时机 |

**缓存布局**:
```
K 缓存: [max_batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
V 缓存: [max_batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
```

### 3. W8A16 量化

仅权重的 INT8 量化，使用 FP16 激活：

```
权重: INT8 [rows, cols]
缩放因子: FP16 [ceil(rows/group_size), cols]
输出: FP16 = dequantize(权重, 缩放因子) @ 激活
```

**优势**:
- 权重显存减少约 50%
- 保持 FP16 激活精度
- 在现代 GPU 上使用高效的 INT8 Tensor Core

### 4. CUDA Kernel

核心操作的优化 Kernel：

| Kernel | 优化技巧 |
|--------|----------|
| W8A16 矩阵乘 | 共享内存 tiling、向量化加载、warp shuffle 归约 |
| 注意力 | Kernel 融合、在线 softmax、内存合并访问 |
| RMSNorm | Warp 级并行归约、向量化内存访问 |
| RoPE | 即时计算、三角函数缓存 |

---

## 数据流

### Prefill 阶段 (Prompt 处理)

```
输入 Tokens ──► Embedding ──► Transformer 层 ──► Logits
                    │              │
                    │              └── KV 缓存写入 (所有位置)
                    │
                    └── 位置 IDs ──► RoPE
```

### Decode 阶段 (Token 生成)

```
单 Token ──► Embedding ──► Transformer 层 ──► Logits ──► 采样
                  │              │                       │
                  │              └── KV 缓存追加         └── 下一 Token
                  │                   + 读取
                  │
                  └── 位置 ID ──► RoPE
```

---

## 内存布局

### 权重存储

```
┌─────────────────────────────────────────────────────────────┐
│                    模型权重                                  │
├─────────────────────────────────────────────────────────────┤
│  Token 嵌入                                                  │
│  [vocab_size, hidden_dim] FP16                               │
├─────────────────────────────────────────────────────────────┤
│  层权重 × N                                                  │
│  ├── 注意力                                                  │
│  │   ├── q_proj:  INT8 [hidden_dim, hidden_dim]              │
│  │   ├── k_proj:  INT8 [hidden_dim, kv_hidden_dim]           │
│  │   ├── v_proj:  INT8 [hidden_dim, kv_hidden_dim]           │
│  │   ├── o_proj:  INT8 [hidden_dim, hidden_dim]              │
│  │   └── 缩放因子: 每层对应的 FP16                           │
│  └── FFN                                                     │
│      ├── gate_proj: INT8 [hidden_dim, intermediate_dim]      │
│      ├── up_proj:   INT8 [hidden_dim, intermediate_dim]      │
│      ├── down_proj: INT8 [intermediate_dim, hidden_dim]      │
│      └── 缩放因子: 每层对应的 FP16                           │
├─────────────────────────────────────────────────────────────┤
│  输出 Norm + LM Head                                         │
└─────────────────────────────────────────────────────────────┘
```

### 激活缓存

```
┌─────────────────────────────────────────────────────────────┐
│                   激活存储                                   │
├─────────────────────────────────────────────────────────────┤
│  隐藏状态: [batch_size, seq_len, hidden_dim] FP16            │
│  注意力输出: [batch_size, num_heads, seq_len, head_dim]      │
│  FFN 中间结果: [batch_size, seq_len, intermediate_dim]       │
└─────────────────────────────────────────────────────────────┘
```

---

## 性能优化

### 1. Kernel 融合

合并操作以减少内存带宽：
- RMSNorm + ResidualAdd
- SiLU + ElementwiseMul (SwiGLU)
- QKV 投影融合 (计划中)

### 2. 内存优化

| 技术 | 收益 |
|------|------|
| W8A16 量化 | 权重显存减少 50% |
| KV 缓存分页 | 高效的变长序列处理 |
| 激活检查点 | 用计算换内存 (计划中) |
| 流并行 | 重叠计算和数据传输 |

### 3. 计算优化

| 技术 | 收益 |
|------|------|
| Warp Shuffle | 减少共享内存使用 |
| 向量化加载 | 更好的内存吞吐 |
| Tensor Core | 加速 INT8/FP16 矩阵乘 |
| 在线 Softmax | 数值稳定性 + 减少遍历次数 |

---

## 设计原则

1. **模块化**：层、Kernel、工具之间清晰分离
2. **类型安全**：使用 `Result<T>` 进行错误处理，强类型
3. **RAII**：CUDA 资源自动管理
4. **可测试性**：全面的单元和属性测试
5. **可扩展性**：易于添加新 Kernel 和采样策略

---

## 未来增强

| 特性 | 状态 | 说明 |
|------|------|------|
| PagedAttention | 计划中 | 支持变长序列的高效批处理 |
| 连续批处理 | 计划中 | 吞吐优化的请求调度 |
| 投机解码 | 评估中 | 通过草稿模型降低延迟 |
| FP8 支持 | 计划中 | 下一代 GPU 量化 |
| 多 GPU | 计划中 | 跨设备的张量并行 |

---

**Languages**: [English](../en/ARCHITECTURE) | [中文](ARCHITECTURE)

[← 返回首页](../../) | [API 参考](API) | [贡献指南](../../CONTRIBUTING)
