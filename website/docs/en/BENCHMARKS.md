---
layout: default
title: "Benchmarks — Tiny-LLM"
description: "Tiny-LLM performance benchmarks and profiling"
nav_order: 6
---

# Benchmarks

Performance benchmarks and profiling data for Tiny-LLM.

---

## Table of Contents

- [System Configuration](#system-configuration)
- [End-to-End Benchmarks](#end-to-end-benchmarks)
- [Kernel Benchmarks](#kernel-benchmarks)
- [Memory Usage](#memory-usage)
- [Profiling Guide](#profiling-guide)

---

## System Configuration

Reference benchmarking system:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A6000 (Ampere, 48 GB) |
| CPU | AMD EPYC 7763 64-Core |
| RAM | 256 GB DDR4 |
| CUDA | 12.2 |
| Driver | 535.104 |

---

## End-to-End Benchmarks

### Throughput (tokens/second)

**Model: 7B parameters, 4096 hidden, 32 layers, 32 heads**

| Batch Size | Sequence Length | Prefill (tok/s) | Decode (tok/s) | Memory (GB) |
|------------|-----------------|-----------------|----------------|-------------|
| 1 | 128 | 12,800 | 85 | 4.2 |
| 1 | 512 | 10,240 | 82 | 5.8 |
| 1 | 2048 | 6,400 | 76 | 11.2 |
| 4 | 128 | 24,000 | 280 | 11.8 |
| 4 | 512 | 18,432 | 270 | 16.4 |

**Note**: Batch > 1 requires sufficient KV cache memory.

### W8A16 vs FP16 Comparison

| Metric | W8A16 | FP16 | Improvement |
|--------|-------|------|-------------|
| Weight Memory | 7.5 GB | 15 GB | **50%** ↓ |
| Activation Memory | Same | Same | - |
| Throughput | 85 tok/s | 78 tok/s | **9%** ↑ |
| Accuracy (perplexity) | 9.12 | 9.08 | 0.4% |

---

## Kernel Benchmarks

### W8A16 Matrix Multiplication

**Configuration: M=1, K=4096, N=4096**

| GPU | Time (μs) | Throughput (TFLOPS) | Tensor Core % |
|-----|-----------|---------------------|---------------|
| RTX A6000 | 42 | 0.80 | 78% |
| A100 | 35 | 0.96 | 82% |
| RTX 4090 | 28 | 1.20 | 85% |

### Attention Decode

**Configuration: batch=1, heads=32, head_dim=128, varying seq_len**

| Seq Len | Time (μs) | Memory Bandwidth (GB/s) |
|---------|-----------|-------------------------|
| 128 | 24 | 420 |
| 512 | 52 | 780 |
| 2048 | 180 | 920 |
| 8192 | 680 | 980 |

**Note**: Decode is memory bandwidth bound due to KV cache reads.

### RMSNorm

| Hidden Dim | Time (μs) | Bandwidth (TB/s) |
|------------|-----------|------------------|
| 4096 | 1.2 | 2.7 |
| 8192 | 2.1 | 3.1 |

---

## Memory Usage

### Model Weights (7B Model)

| Component | W8A16 Size | FP16 Size |
|-----------|------------|-----------|
| Embeddings | 250 MB | 250 MB |
| 32 × Attention Layers | 4.0 GB | 8.0 GB |
| 32 × FFN Layers | 3.5 GB | 7.0 GB |
| Output Norm + LM Head | ~0 | ~0 |
| **Total Weights** | **~7.8 GB** | **~15.3 GB** |

### Runtime Memory

| Configuration | Weights | KV Cache | Activations | Total |
|---------------|---------|----------|-------------|-------|
| Batch=1, Seq=2048 | 7.8 GB | 0.5 GB | 0.1 GB | 8.4 GB |
| Batch=4, Seq=2048 | 7.8 GB | 2.0 GB | 0.4 GB | 10.2 GB |

**KV Cache Formula**: `2 × batch × num_layers × seq_len × num_kv_heads × head_dim × sizeof(half)`

For 7B model (32 layers, 32 heads, 128 head_dim):
- Per token: 2 × 32 × 128 × 2 = 16.4 KB
- 2048 tokens: 32.8 MB per layer → 1.05 GB total per batch

---

## Profiling Guide

### Nsight Compute

Profile individual kernels:

```bash
# Profile specific kernel
ncu --kernel-name attention_decode \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./test_attention

# Full report
ncu -o report.ncu-rep ./benchmark
ncu-ui report.ncu-rep  # Open in GUI
```

### Nsight Systems

Trace full application:

```bash
nsys profile -o profile --stats true ./tiny_llm_demo
nsys-ui profile.qdrep
```

### Custom Timers

```cpp
#include <chrono>

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_;
public:
    Timer() : start_(Clock::now()) {}
    
    float elapsedMs() {
        auto end = Clock::now();
        return std::chrono::duration<float, std::milli>(end - start_).count();
    }
};

// Usage
Timer t;
engine->generate(prompt, config);
std::cout << "Generation took: " << t.elapsedMs() << " ms" << std::endl;
```

---

**Languages**: [English](BENCHMARKS) | [中文](../zh/BENCHMARKS)

[← Developer Guide](DEVELOPER) | [Troubleshooting →](TROUBLESHOOTING)
