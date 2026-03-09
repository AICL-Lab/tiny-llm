---
layout: default
title: Tiny-LLM
---

# Tiny-LLM Inference Engine

轻量级 LLM 推理引擎 — CUDA C++ 实现 W8A16 量化推理，支持 KV Cache 和多种采样策略。

## 核心特性

- **W8A16 量化** — INT8 权重 + FP16 激活，减少 50% 显存占用
- **高效 CUDA Kernel** — 共享内存 tiling、warp shuffle 优化
- **KV Cache 管理** — 支持增量解码，避免重复计算
- **多种采样策略** — 贪婪、温度、top-k、top-p 采样
- **模块化设计** — 易于扩展和定制

## 文档

- [README](README.md) — 项目概述与快速开始
- [API 参考](docs/API.md) — 接口文档

## 快速开始

```bash
# 构建
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# 运行测试
./tiny_llm_tests
```

## 使用示例

```cpp
#include "tiny_llm/inference_engine.h"

using namespace tiny_llm;

ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

auto result = InferenceEngine::load("model.bin", config);
auto engine = std::move(result.value());

// 推理
auto output = engine->generate("Hello", GenerateParams{
    .max_tokens = 100,
    .temperature = 0.7f,
    .top_p = 0.9f
});
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | CUDA C++17 |
| 构建 | CMake 3.18+ |
| GPU | SM 70+ (Volta → Hopper) |
| 量化 | INT8 (W8A16) |

## 链接

- [GitHub 仓库](https://github.com/LessUp/tiny-llm)
- [README](README.md)
