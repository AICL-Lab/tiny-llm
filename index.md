---
layout: default
title: "Tiny-LLM — 轻量级 LLM 推理引擎"
description: "CUDA C++ 实现的 W8A16 量化推理引擎，支持 KV Cache 和多种采样策略"
---

# Tiny-LLM Inference Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

A lightweight LLM inference engine written in CUDA C++. Implements **W8A16 quantization** (INT8 weights + FP16 activations) for 50% memory reduction, with KV Cache for incremental decoding and multiple sampling strategies.

## Key Features

- **W8A16 Quantization** — INT8 weight storage with FP16 activation compute; dequantization happens inside the CUDA kernel
- **Optimized CUDA Kernels** — Shared-memory tiling and warp-shuffle reductions for matmul, attention, and RMSNorm
- **KV Cache** — Pre-allocated GPU memory pool for incremental decoding, avoiding recomputation of past tokens
- **Sampling Strategies** — Greedy, temperature, top-k, and top-p (nucleus) sampling
- **Modular Architecture** — Clean separation of kernels, transformer layers, model loading, and generation logic

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   InferenceEngine                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Model   │  │  Transformer │  │  Generation          │ │
│  │  Loader  │──▶  Layers      │──▶  (Sampling + Decode) │ │
│  └──────────┘  └──────┬───────┘  └──────────────────────┘ │
└────────────────────────┼───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                    CUDA Kernels                            │
│  ┌──────────────┐  ┌──────────┐  ┌───────────────────┐    │
│  │ W8A16 MatMul │  │ Attention│  │ RMSNorm           │    │
│  │ (tiling +    │  │ (KV Cache│  │ (warp shuffle     │    │
│  │  dequant)    │  │  + mask) │  │  reduction)       │    │
│  └──────────────┘  └──────────┘  └───────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

## W8A16 Quantization

```
output = input @ dequant(weight, scales)
       = input(FP16) @ (weight_int8 × scales)
```

Weights are stored as INT8 (1 byte each) instead of FP16 (2 bytes), cutting model memory by **50%**. The per-channel scale factors are stored in FP16. Dequantization is fused into the matmul kernel — no separate dequant pass needed.

## Sampling Strategies

| Strategy | Description |
|----------|-------------|
| **Greedy** | Always pick the highest-probability token |
| **Temperature** | Scale logits by 1/T before softmax; higher T → more random |
| **Top-k** | Sample from the k most probable tokens |
| **Top-p** (nucleus) | Sample from the smallest set whose cumulative probability ≥ p |

## Quick Start

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run tests
./tiny_llm_tests
```

## Usage

```cpp
#include "tiny_llm/inference_engine.h"
using namespace tiny_llm;

ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

auto engine = InferenceEngine::load("model.bin", config).value();

GenerationConfig gen;
gen.max_new_tokens = 100;
gen.temperature = 0.7f;
gen.top_p = 0.9f;
gen.do_sample = true;

auto output = engine->generate({1, 15043, 29892}, gen);  // "Hello,"
```

## Testing

Tests use Google Test + RapidCheck:

```bash
./tiny_llm_tests --gtest_filter="W8A16*"      # Quantized matmul
./tiny_llm_tests --gtest_filter="Attention*"   # Attention mechanism
./tiny_llm_tests --gtest_filter="KVCache*"     # Cache management
```

| Test Suite | What It Covers |
|-----------|----------------|
| W8A16 MatMul | Quantization accuracy, tiling correctness |
| Attention | Masked self-attention, KV cache append |
| RMSNorm | Normalization invariants |
| KV Cache | Allocation, append, multi-sequence |
| Integration | End-to-end prompt → generation |

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | CUDA C++17 |
| Build | CMake 3.18+ |
| GPU | Compute Capability 7.0+ (Volta → Hopper) |
| Quantization | W8A16 (INT8 weights, FP16 activations) |
| Testing | Google Test + RapidCheck |

## 最近更新

| 日期 | 变更 |
|------|------|
| 2026-03-10 | GitHub Pages 优化 — SEO 元数据、kramdown GFM、sparse checkout、changelog 索引 |
| 2026-03-09 | **v2.0.0 重大重构** — KVCache appendKV 层序依赖修复、CMake 现代化 |
| 2026-03-09 | 工作流优化 — GitHub Actions CI、clang-format 检查 |

[查看完整更新日志 →](changelog/)

---

## 文档

- [README](README.md) — 完整项目概述
- [API 参考](docs/API.md) — 公共接口文档
- [更新日志](changelog/) — 版本历史
- [贡献指南](CONTRIBUTING.md) — 如何参与开发

---

[View on GitHub](https://github.com/LessUp/tiny-llm)
