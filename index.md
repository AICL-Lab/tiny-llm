---
layout: default
title: "Tiny-LLM 文档"
description: "Tiny-LLM 文档入口：快速开始、API 参考、更新日志与贡献说明"
---

# Tiny-LLM 文档

Tiny-LLM 的文档首页聚焦在“从哪里开始读”：先帮你完成一次最小构建，再指向 API、更新历史和贡献入口。

## 项目定位

Tiny-LLM 是一个面向学习与工程实验的轻量级 LLM 推理引擎，核心关注点是 W8A16 量化、CUDA Kernel、KV Cache 增量解码，以及模块化的 Transformer 推理链路。

## 适合谁

- 想理解量化推理与 CUDA Kernel 组织方式的学习者
- 想搭建小型推理引擎原型、验证采样与缓存机制的开发者
- 想阅读公共接口、跟踪版本演进并参与维护的贡献者

## 从哪里开始

1. 先看下面的“快速开始”，确认构建环境并跑通测试。
2. 然后阅读 [API 参考](docs/API)，了解 `InferenceEngine`、`KVCacheManager` 与配置结构。
3. 如需了解项目演进和修改背景，继续查看 [更新日志](changelog/)。
4. 准备提交改动时，再阅读 [贡献指南](CONTRIBUTING)。

## 快速开始

### 系统要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 兼容编译器
- NVIDIA GPU（Compute Capability 7.0+）

### 构建与测试

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

## 推荐阅读路径

### 我想先把工程跑起来

- 阅读当前页的“快速开始”
- 再看 [API 参考](docs/API)

### 我想先理解接口和模块边界

- [API 参考](docs/API)
- [贡献指南](CONTRIBUTING)

### 我想了解最近都改了什么

- [更新日志首页](changelog/)
- 最新变更记录见 [2026-03-13 文档与 Pages 规范化](changelog/2026-03-13_docs-information-architecture-standardization)

## 核心文档

| 类别 | 页面 | 说明 |
|------|------|------|
| 概览 | 当前页 | 项目定位、快速开始与阅读路径 |
| 参考 | [API 参考](docs/API) | 公共类型、核心类与 CUDA Kernel 接口 |
| 开发指南 | [贡献指南](CONTRIBUTING) | 开发流程与参与方式 |
| 归档 | [更新日志](changelog/) | 版本历史与调整背景 |
