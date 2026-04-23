---
layout: default
title: "Architecture — Tiny-LLM"
description: "Tiny-LLM system architecture and design documentation"
nav_order: 3
---

# Architecture

System architecture and design documentation for Tiny-LLM inference engine.

---

## Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Memory Layout](#memory-layout)
- [Performance Optimizations](#performance-optimizations)
- [Design Principles](#design-principles)

---

## Overview

Tiny-LLM is a high-performance CUDA C++ inference engine designed for efficient Transformer model inference. It focuses on:

| Feature | Technology | Benefit |
|---------|------------|---------|
| **W8A16 Quantization** | INT8 weights + FP16 activations | ~50% memory reduction |
| **Efficient KV Cache** | Incremental decoding with sequence management | O(1) autoregressive step |
| **Optimized Kernels** | Tensor Core INT8, shared memory tiling | Maximum throughput |
| **Modular Design** | Clean separation of concerns | Easy to extend and test |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        InferenceEngine                               │
├─────────────────────────────────────────────────────────────────────┤
│  Model Loader                                              │
│  ├── Custom binary format parsing                          │
│  ├── Weight dequantization (INT8 → FP16)                   │
│  └── GPU memory allocation                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Transformer Stack (× num_layers)                          │
│  │                                                         │
│  ├─→ Self-Attention Layer                                  │
│  │   ├── RMSNorm                                           │
│  │   ├── QKV Projection (W8A16 MatMul)                     │
│  │   ├── RoPE Position Encoding                            │
│  │   ├── Attention Compute (Decode/Prefill)                │
│  │   ├── Output Projection (W8A16 MatMul)                  │
│  │   └── Residual Connection                               │
│  │                                                         │
│  └─→ FFN Layer                                             │
│      ├── RMSNorm                                           │
│      ├── Gate Projection + SiLU (W8A16)                    │
│      ├── Up Projection (W8A16)                             │
│      ├── Down Projection (W8A16)                           │
│      └── Residual Connection                               │
├─────────────────────────────────────────────────────────────────────┤
│  Output Processing                                         │
│  ├── Final RMSNorm                                         │
│  ├── LM Head Projection                                    │
│  └── Token Sampling (Greedy/Top-k/Top-p)                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. InferenceEngine

The main entry point for model inference.

```cpp
class InferenceEngine {
public:
    // Load model from disk
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& path, const ModelConfig& config);
    
    // Complete generation pipeline
    std::vector<int> generate(
        const std::vector<int>& prompt, 
        const GenerationConfig& config);
    
    // Statistics and profiling
    const GenerationStats& getStats() const;
    void resetStats();
};
```

**Key Responsibilities**:
- Model lifecycle management
- Prefill/decode orchestration
- Token sampling and generation loop
- Performance profiling

### 2. KV Cache Manager

Efficient key-value cache for autoregressive generation.

**Design Motivation**: The v2.0 redesign fixed a critical issue where layer order affected write positions. The new design uses explicit sequence length advancement.

```cpp
class KVCacheManager {
public:
    // Allocate a new sequence slot
    Result<int> allocateSequence(int max_len);
    
    // Append KV for a specific layer (stateless)
    void appendKV(int seq_id, int layer_idx,
                  const half* k, const half* v,
                  int num_tokens, cudaStream_t stream);
    
    // Advance sequence length after all layers
    void advanceSeqLen(int seq_id, int num_tokens);
    
    // Access cached K/V for attention computation
    std::pair<half*, half*> getCache(int seq_id, int layer_idx);
};
```

**Memory Layout**:
```
K Cache: [max_batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
V Cache: [max_batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]
```

### 3. W8A16 Quantization

Weight-only INT8 quantization with FP16 activations.

**Quantization Scheme**:
```
Weight:  INT8 [rows, cols]
Scales:  FP16 [ceil(rows/group_size), cols]
Output:  FP16 = dequantize(Weight, Scales) @ Activation_FP16
```

**Group-wise Quantization**: Weights are divided into groups of 128 elements along the input dimension. Each group shares a scale factor.

```
┌────────────────────────────────────────────────┐
│  QuantizedWeight Layout                        │
├────────────────────────────────────────────────┤
│  weights: int8_t [rows, cols]                  │
│  ├── group 0 (128 elements) → scales[0, :]     │
│  ├── group 1 (128 elements) → scales[1, :]     │
│  └── ...                                       │
│  scales: half [rows/128, cols]                 │
└────────────────────────────────────────────────┘
```

**Benefits**:
- 50% weight memory reduction
- No activation quantization (maintains precision)
- Efficient INT8 Tensor Core utilization on Ampere+

### 4. CUDA Kernel Implementations

#### W8A16 Matrix Multiplication

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

**Optimizations**:
- Shared memory tiling for A matrix
- Vectorized loads (4-byte alignment)
- Warp shuffle for reduction
- Coalesced writeback

#### Attention Kernels

**Decode Attention** (single token vs cached KV):
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

**Prefill Attention** (multiple tokens with causal mask):
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

**Optimizations**:
- Online softmax for numerical stability
- Memory coalescing for cache access
- Kernel fusion opportunities

---

## Data Flow

### Prefill Phase (Prompt Processing)

```
Input Tokens (B, S)
      │
      ▼
┌─────────────┐
│ Embedding   │ (B, S, H)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Transformer Layers (× N)                │
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

### Decode Phase (Token Generation)

```
Single Token (B, 1)
      │
      ▼
┌─────────────┐
│ Embedding   │ (B, 1, H)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Transformer Layers (× N)                │
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

## Memory Layout

### Weight Storage

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Model Weights                                │
├─────────────────────────────────────────────────────────────────────┤
│ Token Embeddings                                                     │
│ [vocab_size, hidden_dim] FP16                                        │
│ ~250 MB (32k vocab, 4k hidden)                                       │
├─────────────────────────────────────────────────────────────────────┤
│ Layer Weights (× num_layers)                                         │
│                                                                      │
│  [Attention]                                                         │
│  ├── q_proj:  INT8 [hidden_dim, hidden_dim]                          │
│  ├── k_proj:  INT8 [hidden_dim, num_kv_heads × head_dim]              │
│  ├── v_proj:  INT8 [hidden_dim, num_kv_heads × head_dim]              │
│  ├── o_proj:  INT8 [hidden_dim, hidden_dim]                          │
│  └── scales: FP16 (various sizes per proj)                           │
│                                                                      │
│  [FFN - SwiGLU]                                                      │
│  ├── gate_proj: INT8 [hidden_dim, intermediate_dim]                  │
│  ├── up_proj:   INT8 [hidden_dim, intermediate_dim]                  │
│  ├── down_proj: INT8 [intermediate_dim, hidden_dim]                  │
│  └── scales: FP16 (various sizes per proj)                           │
│                                                                      │
│  Per-layer: ~500 MB (7B model, 4k/11k dims, W8A16)                   │
│  Total (32 layers): ~16 GB                                           │
├─────────────────────────────────────────────────────────────────────┤
│ Output Head                                                          │
│ [hidden_dim, vocab_size] FP16 (often tied with embeddings)           │
└─────────────────────────────────────────────────────────────────────┘
```

### Activation Buffers

| Buffer | Shape | Dtype | Size (B=1, S=2048, H=4096) |
|--------|-------|-------|---------------------------|
| Hidden States | [B, S, H] | FP16 | 16 MB |
| Attention Output | [B, heads, S, head_dim] | FP16 | 16 MB |
| QKV | [B, S, 3×H] | FP16 | 48 MB |
| FFN Intermediate | [B, S, intermediate_dim] | FP16 | 44 MB |

---

## Performance Optimizations

### Memory Optimizations

| Technique | Implementation | Benefit |
|-----------|----------------|---------|
| W8A16 Quantization | Per-group INT8 weights + FP16 scales | 50% weight memory |
| KV Cache Paging | Pre-allocated with sequence management | Efficient batching |
| Activation Reuse | In-place operations | Reduced allocations |

### Compute Optimizations

| Technique | Application | Speedup |
|-----------|-------------|---------|
| Tensor Cores | INT8 matmul on Ampere+ | 2-4× vs FP16 |
| Warp Shuffle | Reductions | Eliminates shared memory |
| Vectorized Loads | 128-bit memory access | Better bandwidth |
| Kernel Fusion | RMSNorm+Resid, SiLU+Mul | Reduced kernel launch |

### Optimized Kernel List

| Kernel | Optimizations | Throughput |
|--------|---------------|------------|
| `w8a16_matmul` | Tiling, vectorized, warp shuffle | ~80% Tensor Core |
| `attention_decode` | Online softmax, fused KV read | Memory bandwidth bound |
| `attention_prefill` | Tiled softmax, fused causal mask | Compute bound |
| `rmsnorm` | Warp reduction, vectorized | >1TB/s bandwidth |
| `rope` | Cached trig, vectorized | Negligible overhead |

---

## Design Principles

1. **Modularity**: Clear interfaces between layers, kernels, and utilities
2. **Type Safety**: Result<T> for error handling, strong typing throughout
3. **RAII**: Automatic resource management for GPU memory and streams
4. **Testability**: Comprehensive unit tests with property-based testing
5. **Extensibility**: Easy to add new kernels, sampling strategies, model formats

---

**Languages**: [English](ARCHITECTURE) | [中文](../zh/ARCHITECTURE) | [API →](API)

[← Quick Start](QUICKSTART) | [Developer Guide →](DEVELOPER)
