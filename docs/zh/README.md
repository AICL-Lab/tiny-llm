---
layout: default
title: "文档"
description: "Tiny-LLM 文档首页"
nav_order: 0
---

# Tiny-LLM 文档

欢迎来到 Tiny-LLM 文档！Tiny-LLM 是一个轻量级 CUDA/C++ LLM 推理引擎，支持 W8A16 量化和高效的 KV 缓存管理。

---

## 快速开始

刚接触 Tiny-LLM？从这里开始：

| 文档 | 说明 |
|------|------|
| [快速开始](QUICKSTART) | 安装和基础使用 |
| [架构设计](ARCHITECTURE) | 系统设计与组件 |
| [API 参考](API) | 完整 API 文档 |

---

## 核心特性

- **W8A16 量化**: 权重显存减少约 50%
- **KV 缓存**: 高效的增量解码
- **高性能 Kernel**: CUDA 优化实现
- **多种采样**: 贪婪、温度、Top-k、Top-p
- **完整测试**: GoogleTest + RapidCheck

---

## 项目链接

- [GitHub 仓库](https://github.com/LessUp/tiny-llm)
- [更新日志](../../changelog/)
- [贡献指南](../../CONTRIBUTING)

---

## 支持

- [GitHub Issues](https://github.com/LessUp/tiny-llm/issues) 用于 bug 报告
- [GitHub Discussions](https://github.com/LessUp/tiny-llm/discussions) 用于问题讨论

---

**Languages**: [English](../en/) | [中文](./)

[← 返回首页](../../)
