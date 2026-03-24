# Tiny-LLM Inference Engine

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[English](README.md) | 简体中文 | [项目主页](https://lessup.github.io/tiny-llm/)

Tiny-LLM 是一个轻量级 CUDA C++ 推理引擎，用于实践 W8A16 量化、KV Cache 增量解码与模块化 Transformer 推理流程。

当前状态：核心运行时、KV Cache 流程和测试脚手架已经具备，但仓库仍处于实验阶段。默认 demo 二进制当前主要用于报告 CUDA 就绪状态，还不是完整的端到端 CLI；运行时也尚未接通 GGUF 加载。

## 仓库入口

- 提供 INT8 权重 + FP16 激活的 W8A16 量化推理实现
- 包含 matmul、attention、RMSNorm、elementwise 等核心 CUDA Kernel
- 主机端模块覆盖模型加载、Transformer 执行、生成逻辑与 KV Cache 管理
- 详细使用方式、API、更新历史与贡献说明统一维护在文档站中

## 快速开始

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
./tiny_llm_demo
```

说明：
- 配置和构建本项目需要可用的 CUDA Toolkit 与 `nvcc`。
- 当前 demo 主要用于验证 CUDA 可用性并输出运行环境信息。
- `InferenceEngine::load()` 目前只接通了项目测试二进制格式 `ModelLoader::loadBin()`；运行时 `.gguf` 加载尚未打通。

## 接下来读什么

- [文档首页](https://lessup.github.io/tiny-llm/)
- [API 参考](https://lessup.github.io/tiny-llm/docs/API)
- [更新日志](https://lessup.github.io/tiny-llm/changelog/)
- [贡献指南](CONTRIBUTING.md)

## 许可证

MIT License。
