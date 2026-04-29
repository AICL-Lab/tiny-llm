---
layout: default
title: "更新日志 — Tiny-LLM"
description: "Tiny-LLM 版本历史和发布说明"
---

# 更新日志

Tiny-LLM 只保留简洁、面向发布的公开更新日志。这里记录真正对外有意义的版本里程碑。

---

## 语言选择

- **简体中文** (本页)
- [English](index)

---

## 发布版本

### [2.0.2] — 2026-04-27

#### 新增
- **量化工具**: 新增 `quantization.h`/`quantization.cpp`，包含 F32→F16、Q4_0、Q8_0、W8A16 工具函数
- **CLI 增强**: 新增 `--help`、`--version`、`--info` 选项

#### 变更
- **KVCacheManager**: 改用工厂方法 `create()` 实现一致的 `Result<T>` 错误处理
- **代码质量**: 为简单访问器添加 `noexcept`，修复 clang-format-18 违规

#### 基础设施
- CI 简化，使用 `Jimver/cuda-toolkit` action
- 增强 `.clangd` LSP 配置

**详情**: [English](../en/v2.0.2) | [中文](../zh/v2.0.2)

---

### [2.0.1] — 2026-04-16

#### 修复
- **严重**: 测试工具中 `QuantizedWeight` 尺度维度计算错误
- 移除注意力 kernel 中未使用的代码（`q_reg` 数组加载）

**详情**: [English](../en/v2.0.1) | [中文](../zh/v2.0.1)

---

### [2.0.0] — 2026-03-09

#### 变更 ⚠️ 破坏性变更
- **API 重新设计**: KVCache `appendKV()` 现在无状态，需要显式调用 `advanceSeqLen()`
- CMake 现代化，支持 target 导出和架构自动检测

#### 新增
- 包含自动格式检查的 CI 工作流
- `tiny_llm::tiny_llm` CMake 别名 target

**迁移指南**: 任何直接使用 KVCache 的代码都需要在所有层完成后调用 `advanceSeqLen()`。

**详情**: [English](../en/v2.0.0) | [中文](../zh/v2.0.0)

---

## 说明

- 纯基础设施或低信号改动不会进入公开更新日志。
- 如果你想快速了解项目定位与入口，请从 [首页](../) 或 [文档](../docs/) 开始。

---

[← 返回首页](../) | [English Changelog](index) | [贡献指南](../CONTRIBUTING)
