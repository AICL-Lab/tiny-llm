---
layout: default
title: "性能基准 — Tiny-LLM"
description: "Tiny-LLM 性能基准测试和分析"
nav_order: 6
---

# 性能基准

Tiny-LLM 的性能基准测试和分析数据。

---

## 目录

- [系统配置](#系统配置)
- [端到端基准](#端到端基准)
- [Kernel 基准](#kernel-基准)
- [内存使用](#内存使用)
- [性能分析指南](#性能分析指南)

---

## 系统配置

基准测试参考系统：

| 组件 | 规格 |
|-----------|---------------|
| GPU | NVIDIA RTX A6000 (Ampere, 48 GB) |
| CPU | AMD EPYC 7763 64-Core |
| 内存 | 256 GB DDR4 |
| CUDA | 12.2 |
| 驱动 | 535.104 |

---

## 端到端基准

### 吞吐 (tokens/second)

**模型: 7B 参数, 4096 隐藏层, 32 层, 32 头**

| Batch 大小 | 序列长度 | Prefill (tok/s) | Decode (tok/s) | 显存 (GB) |
|------------|-----------------|-----------------|----------------|-------------|
| 1 | 128 | 12,800 | 85 | 4.2 |
| 1 | 512 | 10,240 | 82 | 5.8 |
| 1 | 2048 | 6,400 | 76 | 11.2 |
| 4 | 128 | 24,000 | 280 | 11.8 |
| 4 | 512 | 18,432 | 270 | 16.4 |

**注意**: Batch > 1 需要足够的 KV 缓存显存。

### W8A16 vs FP16 对比

| 指标 | W8A16 | FP16 | 改善 |
|--------|-------|------|-------------|
| 权重显存 | 7.5 GB | 15 GB | **50%** ↓ |
| 激活显存 | 相同 | 相同 | - |
| 吞吐 | 85 tok/s | 78 tok/s | **9%** ↑ |
| 精度 (困惑度) | 9.12 | 9.08 | 0.4% |

---

## Kernel 基准

### W8A16 矩阵乘法

**配置: M=1, K=4096, N=4096**

| GPU | 时间 (μs) | 吞吐 (TFLOPS) | Tensor Core 利用率 |
|-----|-----------|---------------------|---------------|
| RTX A6000 | 42 | 0.80 | 78% |
| A100 | 35 | 0.96 | 82% |
| RTX 4090 | 28 | 1.20 | 85% |

### Attention Decode

**配置: batch=1, heads=32, head_dim=128, 变化 seq_len**

| 序列长度 | 时间 (μs) | 显存带宽 (GB/s) |
|---------|-----------|-------------------------|
| 128 | 24 | 420 |
| 512 | 52 | 780 |
| 2048 | 180 | 920 |
| 8192 | 680 | 980 |

**注意**: Decode 受显存带宽限制，因为需要读取 KV 缓存。

### RMSNorm

| 隐藏维度 | 时间 (μs) | 带宽 (TB/s) |
|------------|-----------|------------------|
| 4096 | 1.2 | 2.7 |
| 8192 | 2.1 | 3.1 |

---

## 内存使用

### 模型权重 (7B 模型)

| 组件 | W8A16 大小 | FP16 大小 |
|-----------|------------|-----------|
| 词嵌入 | 250 MB | 250 MB |
| 32 × 注意力层 | 4.0 GB | 8.0 GB |
| 32 × FFN 层 | 3.5 GB | 7.0 GB |
| 输出 Norm + LM Head | ~0 | ~0 |
| **权重总计** | **~7.8 GB** | **~15.3 GB** |

### 运行时显存

| 配置 | 权重 | KV 缓存 | 激活 | 总计 |
|---------------|---------|----------|-------------|-------|
| Batch=1, Seq=2048 | 7.8 GB | 0.5 GB | 0.1 GB | 8.4 GB |
| Batch=4, Seq=2048 | 7.8 GB | 2.0 GB | 0.4 GB | 10.2 GB |

**KV 缓存公式**: `2 × batch × num_layers × seq_len × num_kv_heads × head_dim × sizeof(half)`

7B 模型示例 (32 层, 32 头, 128 头维度):
- 每 token: 2 × 32 × 128 × 2 = 16.4 KB
- 2048 tokens: 每层 32.8 MB → 每 batch 1.05 GB

---

## 性能分析指南

### Nsight Compute

分析单个 kernel：

```bash
# 分析特定 kernel
ncu --kernel-name attention_decode \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./test_attention

# 完整报告
ncu -o report.ncu-rep ./benchmark
ncu-ui report.ncu-rep  # 用 GUI 打开
```

### Nsight Systems

追踪完整应用：

```bash
nsys profile -o profile --stats true ./tiny_llm_demo
nsys-ui profile.qdrep
```

### 自定义计时器

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

// 使用
Timer t;
engine->generate(prompt, config);
std::cout << "生成耗时: " << t.elapsedMs() << " ms" << std::endl;
```

---

**Languages**: [English](../en/BENCHMARKS) | [中文](BENCHMARKS)

[← 开发者指南](DEVELOPER) | [故障排除 →](TROUBLESHOOTING)
