# Core Architecture Design

> **Version:** 2.0.1 | **Date:** 2026-04-16 | **Status:** Implemented

## Context

This document captures the architectural decisions made during the initial implementation of the Tiny-LLM inference engine.

---

## Goals / Non-Goals

**Goals:**
- Memory efficiency through INT8 weight quantization (~50% reduction)
- Bandwidth optimization via register-level dequantization
- Computational efficiency through warp shuffle and shared memory tiling
- Extensibility for different model architectures

**Non-Goals:**
- Multi-GPU support (single GPU only)
- Training capabilities (inference only)
- Full precision fallback (W8A16 only)

---

## Decisions

### Architecture Overview

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

### Core Components

**Model Loader:**
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

**W8A16 MatMul:**
- Formula: `output = input @ (weight_int8 * scales)`
- Optimization: Shared memory tiling (TILE_M × TILE_N × BLOCK_K)
- Optimization: Warp shuffle reduction
- Optimization: Register-level dequantization

**KV Cache Manager:**
```cpp
Result<int> allocateSequence(int max_len);
void appendKV(seq_id, layer_idx, k, v, n);  // write-only
void advanceSeqLen(seq_id, n);              // explicit advance
void releaseSequence(int seq_id);
```

### Error Handling Pattern

```cpp
// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) throw CudaException(err, __FILE__, __LINE__); \
    } while(0)

// Result type (monadic error handling)
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

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| INT8 precision loss | Validate against FP16 baseline, maintain < 1% error |
| GPU memory fragmentation | Pre-allocated KV cache pool |
| CUDA version compatibility | Support CUDA 11.0+ with forward compatibility |

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
