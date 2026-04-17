# RFC-0001: Tiny-LLM Core Architecture

> **Version:** 2.0.1 | **Date:** 2026-04-16 | **Status:** Implemented

## Overview

This RFC defines the core architecture of the lightweight LLM inference engine, focusing on W8A16 quantized inference and CUDA kernel implementation.

### Design Goals

| Goal | Description |
|------|-------------|
| Memory Efficiency | INT8 weight quantization, reducing ~50% memory |
| Bandwidth Optimization | Register-level dequantization, maximizing bandwidth utilization |
| Computational Efficiency | Warp shuffle + shared memory tiling |
| Extensibility | Modular design supporting different model architectures |

---

## Architecture

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

## Core Components

### Model Loader

```cpp
struct ModelConfig {
    int vocab_size, hidden_dim, num_layers;
    int num_heads, num_kv_heads, head_dim;
    int intermediate_dim, max_seq_len;
    float rope_theta, rms_norm_eps;
};

struct QuantizedWeight {
    int8_t* data;      // INT8 weights
    half* scales;      // Per-group scales
    int rows, cols, group_size;
};
```

### W8A16 MatMul

Formula: `output = input @ (weight_int8 * scales)`

Optimization techniques:
- Shared memory tiling: TILE_M × TILE_N × BLOCK_K
- Warp shuffle reduction
- Register-level dequantization

### KV Cache Manager

Key APIs:
```cpp
Result<int> allocateSequence(int max_len);
void appendKV(seq_id, layer_idx, k, v, n);  // write-only
void advanceSeqLen(seq_id, n);              // explicit advance
void releaseSequence(int seq_id);
```

---

## Correctness Properties

| ID | Property | Validation Requirement |
|----|----------|----------------------|
| P1 | W8A16 relative error < 1% | 2.5, 2.6 |
| P2 | KV Cache invariants | 3.2-3.6 |
| P3 | Causal masking correctness | 4.2 |
| P4 | RMSNorm output RMS ≈ 1 | 4.4 |
| P5 | Incremental decoding equivalent to full computation | 4.6 |
| P6 | Greedy sampling = argmax | 5.2 |
| P7 | Generation length ≤ max_new_tokens | 5.4 |
| P8 | Scale dimension correctness | 1.3, 7.2 |
| P9 | Corrupted file does not crash | 1.5 |

---

## Error Handling

```cpp
// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) throw CudaException(err, __FILE__, __LINE__); \
    } while(0)

// Result type
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

## Test Strategy

| Type | Framework | Coverage |
|------|-----------|----------|
| Unit Tests | GoogleTest | Concrete examples, edge cases |
| Property Tests | RapidCheck | General correctness (100+ iterations) |
| Integration Tests | GoogleTest | End-to-end inference |

---

## File Structure

```
tiny-llm/
├── include/tiny_llm/     # Public headers
│   ├── types.h           # Core types
│   ├── result.h          # Error handling
│   ├── cuda_utils.h      # CUDA utilities
│   ├── kv_cache.h        # KV Cache
│   ├── transformer.h     # Transformer layer
│   ├── model_loader.h    # Model loading
│   └── inference_engine.h
├── kernels/              # CUDA Kernels
│   ├── w8a16_matmul.cu
│   ├── attention.cu
│   ├── rmsnorm.cu
│   └── elementwise.cu
├── src/                  # Implementation
└── tests/                # Tests
```
