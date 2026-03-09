# Tiny-LLM Inference Engine

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[English](README.md) | 简体中文 | [项目主页](https://lessup.github.io/tiny-llm/)

一个轻量级的 LLM 推理引擎，使用 CUDA C++ 实现 W8A16 量化推理。

## 特性

- **W8A16 量化**：INT8 权重 + FP16 激活，减少 50% 显存占用
- **高效 CUDA Kernel**：共享内存 tiling、warp shuffle 优化
- **KV Cache 管理**：支持增量解码，避免重复计算
- **多种采样策略**：贪婪、温度、top-k、top-p 采样
- **模块化设计**：易于扩展和定制
- **工程质量**：CI 流水线、clang-format 格式检查、RAII 内存管理、Result 错误处理

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
└──────────────────────────────────────────────────────────────┘
```

## 系统要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 兼容编译器
- NVIDIA GPU (Compute Capability 7.0+)

## 快速开始

### 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 运行测试

```bash
./tiny_llm_tests
```

### 使用示例

```cpp
#include "tiny_llm/inference_engine.h"

using namespace tiny_llm;

// 加载模型
ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;
// ... 其他配置

auto result = InferenceEngine::load("model.bin", config);
if (result.isErr()) {
    std::cerr << "Failed to load model: " << result.error() << std::endl;
    return 1;
}
auto engine = std::move(result.value());

// 生成文本
std::vector<int> prompt = {1, 15043, 29892};  // "Hello,"
GenerationConfig gen_config;
gen_config.max_new_tokens = 100;
gen_config.temperature = 0.7f;
gen_config.do_sample = true;

auto output = engine->generate(prompt, gen_config);
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
└── CMakeLists.txt             # CMake 构建（v2.0.0，FetchContent GTest）
```

## 核心组件

### W8A16 量化矩阵乘法

使用 INT8 权重和 FP16 激活进行矩阵乘法，在 kernel 内部进行反量化：

```
output = input @ dequant(weight, scales)
       = input @ (weight_int8 * scales)
```

### KV Cache

支持增量解码的 KV Cache 管理器：

- 预分配 GPU 显存池
- 支持多序列并发
- 自动内存回收

### 采样策略

- **贪婪采样**：选择概率最高的 token
- **温度采样**：调整概率分布的锐度
- **Top-k 采样**：只从概率最高的 k 个 token 中采样
- **Top-p 采样**：从累积概率达到 p 的 token 中采样

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

项目使用 Google Test 和 RapidCheck 进行测试：

- **单元测试**：验证各组件的基本功能
- **属性测试**：验证数学性质和不变量
- **集成测试**：验证端到端流程

运行特定测试：

```bash
./tiny_llm_tests --gtest_filter="W8A16*"      # W8A16 测试
./tiny_llm_tests --gtest_filter="Attention*"  # 注意力测试
./tiny_llm_tests --gtest_filter="KVCache*"    # KV Cache 测试
```

## 技术栈

| 类别 | 技术 |
|------|------|
| **语言** | CUDA C++17 |
| **构建** | CMake 3.18+（FetchContent 依赖管理） |
| **GPU** | Compute Capability 7.0+（Volta → Hopper） |
| **量化** | W8A16（INT8 权重 + FP16 激活） |
| **测试** | Google Test v1.14.0 |
| **CI** | GitHub Actions（CUDA 容器构建 + clang-format 检查） |

## 许可证

MIT License
