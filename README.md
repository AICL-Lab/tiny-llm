# Tiny-LLM 推理引擎

> 面向聚焦型 Transformer 工作负载的 CUDA 原生 C++ 推理引擎。

[![CI](https://github.com/AICL-Lab/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/AICL-Lab/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/AICL-Lab/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://aicl-lab.github.io/tiny-llm/)
[![Release](https://img.shields.io/github/v/release/AICL-Lab/tiny-llm?include_prereleases&label=version)](https://github.com/AICL-Lab/tiny-llm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[文档](https://aicl-lab.github.io/tiny-llm/) • [架构说明](https://aicl-lab.github.io/tiny-llm/architecture/) • [API](https://aicl-lab.github.io/tiny-llm/api/) • [路线图](ROADMAP.md) • [更新日志](CHANGELOG.md)

---

## 项目概述

Tiny-LLM 将仓库表面保持得尽量小：CUDA/C++17 内核、W8A16 量化、显式 KV Cache 管理，以及一条更容易审计和维护的精简运行时路径。

在五仓学习路径中，本仓库只负责模型权重到 token 生成的运行时主线；CUDA/Triton kernel 学习和 Serving 调度保持在各自主仓。整体顺序见 [`cuda-kernel-academy/LEARNING_PATH.md`](https://github.com/AICL-Lab/cuda-kernel-academy/blob/master/LEARNING_PATH.md)。

## 项目状态（诚实声明）

| 能力 | 状态 |
|------|------|
| W8A16 量化 kernel 与运行时路径 | ✅ 已实现，有差分测试 |
| KV Cache 管理、采样（temperature/top-k/top-p） | ✅ 已实现 |
| GGUF 解析（F16/F32/Q4_0/Q8_0 反量化） | ✅ 已实现，合成数据测试通过 |
| tokenizer | ❌ 未实现（generate API 以 token id 为输入输出） |
| 真实模型端到端验证 | ❌ 未完成 |
| 端到端性能基准 | ❌ 未完成，暂无可复现的性能数字 |

当前开发重点见 [ROADMAP](ROADMAP.md)。性能相关的文档只描述方法与计划，不引用未实测的数字。

## 已实现能力

- **W8A16 推理路径**：INT8 权重 + FP16 激活
- **显式 KV Cache 管理**：面向自回归解码
- **CUDA 原生 Kernel**：共享内存与 warp 级优化模式
- **基于 `Result<T>` 的可失败 API**：宿主侧错误传播更直接
- **GoogleTest 测试覆盖**：kernel 数值差分、KV Cache 不变量、模型加载等核心路径

## 模型加载

- `InferenceEngine::load()` 支持 **GGUF** 和**二进制运行时格式**两种加载路径。
- GGUF 路径：`GGUFParser` 解析文件、提取模型配置，读取 tensor 数据并反量化（支持 F16/F32/Q4_0/Q8_0），重量化为 W8A16 后上传 GPU。
- 二进制路径：`loadBin()` 直接读取预量化的 W8A16 权重，主要用于测试。

> 注意：GGUF 加载路径已实现并通过合成数据测试，但**尚未用真实模型文件验证**；CLI（`tiny_llm_demo`）目前只报告 CUDA 就绪状态与格式说明，端到端生成请使用库 API。

## 从源码构建

Tiny-LLM 需要可用的 CUDA 工具链（`nvcc` 在 `PATH` 中，或已正确配置 CUDA 安装）。

| 组件 | 最低要求 |
|---|---|
| NVIDIA GPU | 计算能力 7.0+ |
| CUDA Toolkit | 11.0+ |
| CMake | 3.18+ |
| C++ 编译器 | GCC 9+ 或 Clang 10+ |

```bash
git clone https://github.com/AICL-Lab/tiny-llm.git
cd tiny-llm

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
```

## 最小使用示例

```cpp
#include <iostream>
#include <tiny_llm/inference_engine.h>

int main() {
    using namespace tiny_llm;

    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;

    auto engine_result = InferenceEngine::load("model.bin", config);
    if (engine_result.isErr()) {
        std::cerr << engine_result.error() << '\n';
        return 1;
    }

    GenerationConfig gen;
    gen.max_new_tokens = 64;
    gen.temperature = 0.7f;
    gen.top_p = 0.9f;
    gen.do_sample = true;

    auto engine = std::move(engine_result.value());
    auto output = engine->generate({1, 15043, 29892}, gen);
    if (output.isErr()) {
        std::cerr << output.error() << '\n';
        return 1;
    }

    return 0;
}
```

## 仓库结构

```text
include/tiny_llm/         公共头文件
src/                      主机端 C++ 实现
kernels/                  CUDA kernels
tests/                    单元测试与属性测试
docs/                     VitePress 文档站点
.github/workflows/        CI、Pages、release 自动化
CHANGELOG.md              唯一的已跟踪发布历史
```

## 参与贡献

欢迎提交 Issue 与 Pull Request。请保持改动聚焦，让文档与真实运行时边界一致，并避免重新引入重复的流程脚手架。详见[开发者指南](https://aicl-lab.github.io/tiny-llm/contributing/)。

## 许可证

Tiny-LLM 采用 [MIT License](LICENSE)。
