# Tiny-LLM Inference Engine

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[English](README.md) | 简体中文 | [项目主页](https://lessup.github.io/tiny-llm/)

Tiny-LLM 是一个轻量级 CUDA C++ 推理引擎，用于实践 W8A16 量化、KV Cache 增量解码与模块化 Transformer 推理流程。

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
```

## 接下来读什么

- [文档首页](https://lessup.github.io/tiny-llm/)
- [API 参考](https://lessup.github.io/tiny-llm/docs/API)
- [更新日志](https://lessup.github.io/tiny-llm/changelog/)
- [贡献指南](CONTRIBUTING.md)

## 许可证

MIT License。
