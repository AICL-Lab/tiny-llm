---
layout: default
title: "Architecture — Tiny-LLM"
description: "Tiny-LLM system architecture and design principles"
nav_order: 2
---

# Architecture

This document describes the system architecture and design principles of Tiny-LLM.

---

## Overview

Tiny-LLM is a lightweight CUDA/C++ inference engine designed for:
- **W8A16 quantization**: INT8 weights with FP16 activations for ~50% memory reduction
- **Efficient KV Cache**: Incremental decoding with sequence management
- **High-performance kernels**: Optimized CUDA kernels with shared memory tiling and warp shuffle

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    InferenceEngine                           │
├─────────────────────────────────────────────────────────────┤
│  Model Loader ──► Weights (INT8 + FP16 scales)              │
├─────────────────────────────────────────────────────────────┤
│  Transformer Layers × N                                     │
│  ├── Attention: W8A16 MatMul + KV Cache + RoPE + Causal Mask│
│  │   ├── Q, K, V projections (W8A16)                        │
│  │   ├── RoPE position encoding                             │
│  │   ├── Attention computation (decode/prefill)            │
│  │   └── Output projection (W8A16)                          │
│  └── FFN: W8A16 MatMul + SwiGLU                             │
│      ├── Gate projection + SiLU                             │
│      └── Up projection multiplication                       │
├─────────────────────────────────────────────────────────────┤
│  RMSNorm + Residual Connections                             │
├─────────────────────────────────────────────────────────────┤
│  Sampling: Greedy / Temperature / Top-k / Top-p             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Inference Engine

The `InferenceEngine` is the main entry point for inference:

| Capability | Description |
|------------|-------------|
| Model Loading | Supports custom binary format (GGUF support planned) |
| Token Generation | Provides both prefill and decode phases |
| Sampling Strategies | Greedy, temperature, top-k, top-p |
| Statistics | Performance metrics and memory usage tracking |

### 2. KV Cache Manager

Efficient key-value cache for incremental decoding:

| Feature | Description |
|---------|-------------|
| Pre-allocated Memory | All cache memory allocated upfront |
| Sequence Management | Dynamic allocation/release of sequences |
| Stateless Append | Write position independent of layer order |
| Explicit Advance | Caller controls when to advance sequence length |

**Cache Layout**:
```
K Cache: [max_batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
V Cache: [max_batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
```

### 3. W8A16 Quantization

Weight-only INT8 quantization with FP16 activations:

```
Weight: INT8 [rows, cols]
Scales: FP16 [ceil(rows/group_size), cols]
Output: FP16 = dequantize(W, S) @ Activation
```

**Benefits**:
- ~50% weight memory reduction
- FP16 activation precision maintained
- Efficient INT8 tensor cores on modern GPUs

### 4. CUDA Kernels

Optimized kernels for core operations:

| Kernel | Optimization Techniques |
|--------|------------------------|
| W8A16 MatMul | Shared memory tiling, vectorized loads, warp shuffle reduction |
| Attention | Kernel fusion, online softmax, memory coalescing |
| RMSNorm | Warp-level parallel reduction, vectorized memory access |
| RoPE | On-the-fly computation, trigonometric caching |

---

## Data Flow

### Prefill Phase (Prompt Processing)

```
Input Tokens ──► Embedding ──► Transformer Layers ──► Logits
                    │              │
                    │              └── KV Cache Write (all positions)
                    │
                    └── Position IDs ──► RoPE
```

### Decode Phase (Token Generation)

```
Single Token ──► Embedding ──► Transformer Layers ──► Logits ──► Sampling
                     │              │                          │
                     │              └── KV Cache Append         └── Next Token
                     │                   + Read
                     │
                     └── Position ID ──► RoPE
```

---

## Memory Layout

### Weight Storage

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Weights                             │
├─────────────────────────────────────────────────────────────┤
│  Token Embeddings                                            │
│  [vocab_size, hidden_dim] FP16                               │
├─────────────────────────────────────────────────────────────┤
│  Layer Weights × N                                           │
│  ├── Attention                                               │
│  │   ├── q_proj:  INT8 [hidden_dim, hidden_dim]              │
│  │   ├── k_proj:  INT8 [hidden_dim, kv_hidden_dim]           │
│  │   ├── v_proj:  INT8 [hidden_dim, kv_hidden_dim]           │
│  │   ├── o_proj:  INT8 [hidden_dim, hidden_dim]              │
│  │   └── scales: FP16 for each projection                    │
│  └── FFN                                                     │
│      ├── gate_proj: INT8 [hidden_dim, intermediate_dim]      │
│      ├── up_proj:   INT8 [hidden_dim, intermediate_dim]      │
│      ├── down_proj: INT8 [intermediate_dim, hidden_dim]      │
│      └── scales: FP16 for each projection                    │
├─────────────────────────────────────────────────────────────┤
│  Output Norm + LM Head                                       │
└─────────────────────────────────────────────────────────────┘
```

### Activation Buffers

```
┌─────────────────────────────────────────────────────────────┐
│                   Activation Storage                         │
├─────────────────────────────────────────────────────────────┤
│  Hidden States: [batch_size, seq_len, hidden_dim] FP16       │
│  Attention Output: [batch_size, num_heads, seq_len, head_dim]│
│  FFN Intermediate: [batch_size, seq_len, intermediate_dim]   │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Optimizations

### 1. Kernel Fusion

Combined operations to reduce memory bandwidth:
- RMSNorm + ResidualAdd
- SiLU + ElementwiseMul (SwiGLU)
- QKV projection fusion (planned)

### 2. Memory Optimization

| Technique | Benefit |
|-----------|---------|
| W8A16 Quantization | 50% weight memory reduction |
| KV Cache Paged | Efficient variable-length sequences |
| Activation Checkpointing | Trade compute for memory (planned) |
| Stream Parallelism | Overlap compute and data transfer |

### 3. Compute Optimization

| Technique | Benefit |
|-----------|---------|
| Warp Shuffle | Reduce shared memory usage |
| Vectorized Loads | Better memory throughput |
| Tensor Cores | Accelerated INT8/FP16 matmul |
| Online Softmax | Numerical stability + fewer passes |

---

## Design Principles

1. **Modularity**: Clear separation between layers, kernels, and utilities
2. **Type Safety**: `Result<T>` for error handling, strong typing
3. **RAII**: Automatic resource management for CUDA resources
4. **Testability**: Comprehensive unit and property-based tests
5. **Extensibility**: Easy to add new kernels and sampling strategies

---

## Future Enhancements

| Feature | Status | Description |
|---------|--------|-------------|
| PagedAttention | Planned | Efficient batching with variable-length sequences |
| Continuous Batching | Planned | Throughput-optimized request scheduling |
| Speculative Decoding | Evaluating | Latency reduction via draft model |
| FP8 Support | Planned | Next-gen GPU quantization |
| Multi-GPU | Planned | Tensor parallelism across devices |

---

**Languages**: [English](ARCHITECTURE) | [中文](../zh/ARCHITECTURE)

[← Home](../../) | [API Reference](API) | [Contributing](../../CONTRIBUTING)
