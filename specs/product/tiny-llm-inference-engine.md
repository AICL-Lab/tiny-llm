# Tiny-LLM Inference Engine — Product Specification

> **Version:** 2.0.1 | **Date:** 2026-04-16 | **Status:** Implemented

## Overview

A lightweight LLM inference engine implementing W8A16 quantization, KV Cache incremental decoding, and modular Transformer inference for CUDA-enabled GPUs.

---

## Terminology

| Term | Description |
|------|-------------|
| W8A16 | Mixed precision: INT8 weights, FP16 activations |
| KV Cache | Key-Value cache for autoregressive generation |
| Warp Shuffle | CUDA warp-level data exchange instruction |
| GQA | Grouped Query Attention |
| GGUF | LLM model weight file format |

---

## Functional Requirements

### REQ-1: Model Loading

- [x] Parse GGUF file headers and configuration
- [x] Read binary format quantized weights
- [x] Load INT8 weights and scales simultaneously
- [x] Allocate GPU memory and transfer data
- [x] Return error for corrupted files
- [x] Provide structured model representation

### REQ-2: W8A16 Matrix Multiplication

- [x] Accept INT8 weights, FP16 activations, and scales
- [x] Register-level dequantization
- [x] Use CUDA intrinsics
- [x] Warp shuffle optimization
- [x] Configurable tile sizes
- [x] Error vs FP16 baseline < 1%

### REQ-3: KV Cache Management

- [x] Pre-allocated GPU memory pool
- [x] Allocate/release sequence slots
- [x] Append KV pairs incrementally
- [x] Track per-sequence usage
- [x] Return error on memory exhaustion

### REQ-4: Transformer Layer

- [x] Q, K, V projections using W8A16
- [x] Causal masking
- [x] FFN using W8A16
- [x] RMSNorm
- [x] Residual connections
- [x] Incremental decoding

### REQ-5: Token Generation

- [x] Prefill parallel processing
- [x] Compute logits + sampling
- [x] Update KV cache
- [x] Configurable maximum length
- [x] EOS detection for stopping
- [x] Tokens/second statistics

### REQ-6: Performance Optimization

- [x] Global memory coalesced access
- [x] Shared memory tiling
- [x] Optimized thread block dimensions
- [x] Minimized synchronization points
- [x] CUDA streams support

### REQ-7: Error Handling

- [x] CUDA error capture
- [x] Weight dimension validation
- [x] Numerical validation options
- [x] Memory allocation failure reporting
- [x] Profiling mode timing

---

## Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Precision | W8A16 achieves 99%+ precision vs FP16 baseline |
| Memory | INT8 reduces ~50% weight memory |
| Compatibility | CUDA 11.0+, CMake 3.18+, C++17 |
| GPU | Compute Capability 7.0+ |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.1 | 2026-04-16 | Fixed scale dimension calculation bug |
| 2.0.0 | 2026-03-09 | KVCache API refactoring |
| 1.0.0 | 2025-02-13 | Initial release |
