# Tiny-LLM Inference Engine

[English](README.md) | 简体中文

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

一个轻量级的 LLM 推理引擎，使用 CUDA C++ 实现 W8A16 量化推理。

## 特性

- **W8A16 量化**：INT8 权重 + FP16 激活，减少 50% 显存占用
- **高效 CUDA Kernel**：共享内存 tiling、warp shuffle 优化
- **KV Cache 管理**：支持增量解码，避免重复计算
- **多种采样策略**：贪婪、温度、top-k、top-p 采样
- **模块化设计**：易于扩展和定制

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
├── include/tiny_llm/     # 头文件
│   ├── types.h           # 数据类型定义
│   ├── result.h          # 错误处理
│   ├── cuda_utils.h      # CUDA 工具
│   ├── kv_cache.h        # KV Cache 管理
│   ├── transformer.h     # Transformer 层
│   └── inference_engine.h # 推理引擎
├── kernels/              # CUDA Kernel
│   ├── w8a16_matmul.cu   # W8A16 矩阵乘法
│   ├── attention.cu      # 注意力计算
│   └── rmsnorm.cu        # RMSNorm 归一化
├── src/                  # 源文件
│   ├── kv_cache.cpp
│   ├── transformer.cpp
│   ├── model_loader.cpp
│   └── inference_engine.cpp
├── tests/                # 测试文件
└── .kiro/specs/          # 设计文档
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

- 共享内存 tiling 减少全局内存访问
- Warp shuffle 进行高效规约
- 内存合并访问优化
- CUDA Stream 并行

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

## 许可证

MIT License
