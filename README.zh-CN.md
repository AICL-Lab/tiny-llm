# Tiny-LLM 推理引擎

> 面向聚焦型 Transformer 部署场景的 CUDA 原生 C++ 推理引擎。

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![Release](https://img.shields.io/github/v/release/LessUp/tiny-llm?include_prereleases&label=version)](https://github.com/LessUp/tiny-llm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[English](README.md) • [简体中文](README.zh-CN.md) • [文档](https://lessup.github.io/tiny-llm/) • [架构说明](https://lessup.github.io/tiny-llm/docs/zh/ARCHITECTURE) • [更新日志](https://lessup.github.io/tiny-llm/changelog/)

---

## 项目概述

Tiny-LLM 是一个面向 NVIDIA GPU 的轻量级 Transformer 推理引擎，围绕一套聚焦的 CUDA/C++ 技术栈构建：W8A16 量化、显式 KV Cache 管理、手工优化的 CUDA Kernel，以及用来约束仓库演进的 OpenSpec 工作流。

## 为什么是 Tiny-LLM

- **W8A16 量化路径**：INT8 权重 + FP16 激活
- **显式 KV Cache 管理**：适合增量解码
- **CUDA 原生 Kernel**：共享内存与 warp 级优化模式
- **运行时依赖面很小**：主要运行时依赖为 `spdlog`
- **OpenSpec 驱动开发**：让架构、文档和改动保持同步

## 核心能力

| 能力 | 说明 | 状态 |
|---|---|---|
| W8A16 推理路径 | 量化权重、FP16 激活、CUDA Kernel | 稳定 |
| KV Cache 管理 | 序列分配、增长和释放 | 稳定 |
| 采样工具 | Greedy、temperature、top-k、top-p | 稳定 |
| 错误处理 | 基于 `Result<T>` 的可失败 API | 稳定 |
| 测试策略 | GoogleTest + RapidCheck 覆盖核心路径 | 持续维护 |

## 从源码构建

Tiny-LLM 需要可用的 CUDA 工具链（`nvcc` 在 `PATH` 中，或已正确配置的 CUDA 安装）。

| 组件 | 最低要求 |
|---|---|
| NVIDIA GPU | 计算能力 7.0+ |
| CUDA Toolkit | 11.0+ |
| CMake | 3.18+ |
| C++ 编译器 | GCC 9+ 或 Clang 10+ |

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
```

## 最小使用示例

```cpp
#include <tiny_llm/inference_engine.h>

int main() {
    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;

    auto engine = InferenceEngine::load("model.bin", config).value();

    GenerationConfig gen;
    gen.max_new_tokens = 256;
    gen.temperature = 0.7f;
    gen.top_p = 0.9f;

    auto output = engine.generate({1, 15043, 29892}, gen);
    (void)output;
}
```

## 仓库结构

```text
openspec/                 当前规范、活跃变更、归档变更、schemas
include/tiny_llm/         公共头文件
src/                      主机端 C++ 实现
kernels/                  CUDA kernels
tests/                    单元测试与属性测试
website/                  GitHub Pages 站点与公开文档
.github/workflows/        CI、Pages、release 自动化
```

## 文档与规范

- **项目主页：** https://lessup.github.io/tiny-llm/
- **架构说明：** https://lessup.github.io/tiny-llm/docs/zh/ARCHITECTURE
- **开发者指南：** https://lessup.github.io/tiny-llm/docs/zh/DEVELOPER
- **OpenSpec 规范源：** `openspec/specs/`
- **变更历史：** `openspec/changes/`

## 参与贡献

Tiny-LLM 对非平凡改动采用 OpenSpec-first 工作流。

- 先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)
- 再阅读仓库级约束 [AGENTS.md](AGENTS.md)
- 通过 `/opsx:propose`、`/opsx:apply`、`/opsx:archive` 进行结构化开发

## 许可证

Tiny-LLM 采用 [MIT License](LICENSE)。
