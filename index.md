---
layout: default
title: "Tiny-LLM 文档"
description: "轻量级 LLM 推理引擎：W8A16 量化、KV Cache、CUDA Kernel"
---

# Tiny-LLM

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**轻量级 LLM 推理引擎** — 专注于 W8A16 量化、KV Cache 增量解码与 CUDA Kernel 实现。

---

## 特性

| 特性 | 描述 |
|------|------|
| **W8A16 量化** | INT8 权重 + FP16 激活，显存减少 ~50% |
| **KV Cache** | 增量解码，高效序列管理 |
| **高性能 Kernel** | Shared memory tiling, Warp shuffle |
| **多采样策略** | Greedy, Temperature, Top-k, Top-p |
| **完整测试** | GoogleTest + RapidCheck 属性测试 |

---

## 快速开始

### 系统要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 编译器
- NVIDIA GPU (SM 7.0+)

### 构建

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 测试

```bash
ctest --output-on-failure
./tiny_llm_demo
```

---

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                   InferenceEngine                        │
├─────────────────────────────────────────────────────────┤
│  Model Loader ──► Weights (INT8 + FP16 scales)          │
├─────────────────────────────────────────────────────────┤
│  Transformer Layers × N                                 │
│  ├── Attention: W8A16 MatMul + KV Cache + Causal Mask   │
│  └── FFN: W8A16 MatMul + SwiGLU                         │
├─────────────────────────────────────────────────────────┤
│  Sampling: Greedy / Temperature / Top-k / Top-p         │
└─────────────────────────────────────────────────────────┘
```

---

## 文档

| 页面 | 说明 |
|------|------|
| [API 参考](docs/API) | 公共接口、核心类、CUDA Kernel API |
| [贡献指南](CONTRIBUTING) | 开发流程、代码规范、提交格式 |
| [更新日志](changelog/) | 版本历史与变更记录 |

---

## 版本

**v2.0.1** (2026-04-16)

- 修复 QuantizedWeight scale 尺寸计算 bug
- 移除 attention kernel 冗余代码

详见 [更新日志](changelog/)

---

## 许可证

[MIT License](LICENSE)
