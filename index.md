---
layout: default
title: "Tiny-LLM — 轻量级 LLM 推理引擎"
description: "CUDA C++ 实现的 W8A16 量化推理引擎，支持 KV Cache、多种采样策略和模块化 Transformer 架构"
---

# Tiny-LLM Inference Engine

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

一个轻量级的 LLM 推理引擎，使用 CUDA C++ 实现。支持 **W8A16 量化**（INT8 权重 + FP16 激活）将显存占用降低 50%，内置 KV Cache 增量解码和多种采样策略。

## 核心特性

- **W8A16 量化推理** — INT8 权重存储 + FP16 激活计算，反量化在 CUDA kernel 内融合完成
- **高效 CUDA Kernel** — 共享内存 tiling、warp shuffle 规约，覆盖 matmul / attention / RMSNorm
- **KV Cache 管理** — GPU 显存池预分配，支持增量解码和多序列并发，自动内存回收
- **多种采样策略** — 贪婪、温度、top-k、top-p (nucleus) 采样
- **模块化架构** — Kernel / Transformer 层 / 模型加载 / 生成逻辑清晰分离
- **工程质量** — CI 流水线、clang-format 格式检查、RAII 内存管理、Result 错误处理

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    InferenceEngine                           │
│  ┌───────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │  Model    │  │  Transformer  │  │  Generation          │ │
│  │  Loader   │──▶  Layers       │──▶  (Sampling + Decode) │ │
│  └───────────┘  └───────┬───────┘  └──────────────────────┘ │
│                         │                                    │
│  ┌───────────┐  ┌───────▼───────┐  ┌──────────────────────┐ │
│  │  Stream   │  │  KV Cache     │  │  Result<T>           │ │
│  │  Pool     │  │  Manager      │  │  Error Handling      │ │
│  └───────────┘  └───────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     CUDA Kernels                             │
│  ┌──────────────┐  ┌───────────┐  ┌────────────────────┐    │
│  │ W8A16 MatMul │  │ Attention │  │ RMSNorm            │    │
│  │ (tiling +    │  │ (KV Cache │  │ (warp shuffle      │    │
│  │  dequant)    │  │  + mask)  │  │  reduction)        │    │
│  └──────────────┘  └───────────┘  └────────────────────┘    │
│  ┌──────────────┐  ┌───────────┐                             │
│  │ Elementwise  │  │ Warp      │                             │
│  │ (SiLU, add)  │  │ Utilities │                             │
│  └──────────────┘  └───────────┘                             │
└──────────────────────────────────────────────────────────────┘
```

## W8A16 量化原理

```
output = input @ dequant(weight, scales)
       = input(FP16) @ (weight_int8 × scales)
```

权重以 INT8（1 字节）存储，相比 FP16（2 字节）减少 **50%** 显存占用。Per-channel scale 因子以 FP16 存储，反量化直接融合在 matmul kernel 内部，无需额外的 dequant pass。

## 采样策略

| 策略 | 说明 |
|------|------|
| **贪婪采样** | 始终选择概率最高的 token |
| **温度采样** | 以 1/T 缩放 logits 后 softmax；T 越大越随机 |
| **Top-k** | 仅从概率最高的 k 个 token 中采样 |
| **Top-p** (nucleus) | 从累积概率达到 p 的最小 token 集合中采样 |

## 快速开始

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 运行测试
ctest --output-on-failure
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
if (result.isErr()) {
    std::cerr << "Error: " << result.error() << std::endl;
    return 1;
}
auto engine = std::move(result.value());

GenerationConfig gen;
gen.max_new_tokens = 100;
gen.temperature = 0.7f;
gen.top_p = 0.9f;
gen.do_sample = true;

auto output = engine->generate({1, 15043, 29892}, gen);  // "Hello,"
```

## 项目结构

```
tiny-llm/
├── include/tiny_llm/          # 公共头文件
│   ├── types.h                # 数据类型 (ModelConfig, GenerationConfig, QuantizedWeight)
│   ├── result.h               # Result<T> 错误处理（类 Rust 风格）
│   ├── cuda_utils.h           # CUDA 错误检查、DeviceBuffer<T> RAII
│   ├── cuda_streams.h         # StreamPool CUDA 流池
│   ├── kv_cache.h             # KVCacheManager 管理器
│   ├── model_loader.h         # 模型加载器
│   ├── transformer.h          # TransformerLayer 层
│   └── inference_engine.h     # InferenceEngine 推理引擎
├── kernels/                   # CUDA Kernel 实现
│   ├── w8a16_matmul.cu/.cuh   # W8A16 量化矩阵乘法（tiling + 融合反量化）
│   ├── attention.cu/.cuh      # 注意力计算（prefill + decode，KV Cache 集成）
│   ├── rmsnorm.cu/.cuh        # RMSNorm 归一化（warp shuffle 规约）
│   ├── elementwise.cu/.cuh    # 逐元素运算（SiLU 激活、残差加法）
│   └── warp_utils.cuh         # Warp 级原语工具
├── src/                       # 主机端源文件
│   ├── inference_engine.cpp   # 推理引擎主逻辑
│   ├── transformer.cpp        # Transformer 层前向传播
│   ├── kv_cache.cpp           # KV Cache 分配 / 追加 / 回收
│   ├── model_loader.cpp       # 模型文件加载与权重初始化
│   └── main.cpp               # Demo 入口
├── tests/                     # 测试（Google Test）
│   ├── test_kernels.cu        # Kernel 正确性测试
│   ├── test_w8a16_matmul.cu   # W8A16 量化精度测试
│   ├── test_kv_cache.cpp      # KV Cache 管理测试
│   ├── test_transformer.cu    # Transformer 层测试
│   ├── test_inference_engine.cu # 推理引擎测试
│   └── test_integration.cu    # 端到端集成测试
└── CMakeLists.txt             # CMake 构建（v2.0.0，FetchContent GTest）
```

## 性能优化

| 优化技术 | 说明 |
|----------|------|
| **共享内存 Tiling** | W8A16 matmul 和 attention 使用 tile 分块减少全局内存访问 |
| **Warp Shuffle 规约** | RMSNorm 使用 `__shfl_xor_sync` 替代共享内存规约 |
| **融合反量化** | INT8→FP16 反量化在 matmul kernel 内完成，零额外 pass |
| **内存合并访问** | Kernel 数据布局优化，保证 warp 内线程连续访问全局内存 |
| **KV Cache 预分配** | GPU 显存池一次性分配，避免推理过程中的 cudaMalloc 开销 |
| **CUDA Stream 并行** | StreamPool 支持多流并发执行，提升 GPU 利用率 |

## GPU 架构支持

| 架构 | Compute Capability | 代表显卡 |
|------|-------------------|----------|
| Volta | SM 7.0 | V100 |
| Turing | SM 7.5 | RTX 2080, T4 |
| Ampere | SM 8.0 / 8.6 | A100, RTX 3090 |
| Ada Lovelace | SM 8.9 | RTX 4090, L40 |
| Hopper | SM 9.0 | H100 |

## 测试

项目使用 Google Test 进行全面测试：

```bash
./tiny_llm_tests --gtest_filter="W8A16*"       # W8A16 量化矩阵乘法
./tiny_llm_tests --gtest_filter="Attention*"    # 注意力机制
./tiny_llm_tests --gtest_filter="KVCache*"      # KV Cache 管理
./tiny_llm_tests --gtest_filter="Integration*"  # 端到端集成
```

| 测试套件 | 覆盖内容 |
|----------|----------|
| **W8A16 MatMul** | 量化精度、tiling 正确性、边界尺寸 |
| **Attention** | Masked self-attention、KV cache 追加、prefill/decode |
| **RMSNorm** | 归一化不变量、数值稳定性 |
| **KV Cache** | 分配 / 追加 / 多序列 / advanceSeqLen |
| **Transformer** | 层前向传播、权重加载 |
| **Integration** | 端到端 prompt → generation 流程 |

## 技术栈

| 类别 | 技术 |
|------|------|
| **语言** | CUDA C++17 |
| **构建** | CMake 3.18+（FetchContent 依赖管理） |
| **GPU** | Compute Capability 7.0+（Volta → Hopper） |
| **量化** | W8A16（INT8 权重 + FP16 激活） |
| **测试** | Google Test v1.14.0 |
| **CI** | GitHub Actions（CUDA 容器构建 + clang-format 检查） |
| **代码风格** | clang-format + .editorconfig |

## 最近更新

| 日期 | 变更 |
|------|------|
| 2026-03-10 | GitHub Pages 完善 — 文档 frontmatter、导航页脚、项目结构补全 |
| 2026-03-10 | GitHub Pages 优化 — SEO 元数据、kramdown GFM、sparse checkout |
| 2026-03-09 | **v2.0.0 重大重构** — KVCache appendKV 层序依赖修复、CMake 现代化 |
| 2026-03-09 | 工作流优化 — GitHub Actions CI、clang-format 检查 |

[查看完整更新日志 →](changelog/)

---

## 文档

- [README](README.md) — 项目概述（English）
- [README 中文](README.zh-CN.md) — 项目概述（简体中文）
- [API 参考](docs/API) — 公共接口文档
- [更新日志](changelog/) — 版本历史
- [贡献指南](CONTRIBUTING) — 如何参与开发

---

[View on GitHub](https://github.com/LessUp/tiny-llm)
