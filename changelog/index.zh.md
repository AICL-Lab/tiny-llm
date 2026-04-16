---
layout: default
title: "更新日志 — Tiny-LLM"
description: "Tiny-LLM 版本历史和发布说明"
---

# 更新日志

本文档记录项目的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

---

## 语言选择

- **简体中文** (本页)
- [English](index)

---

## 发布历史

### [未发布]

- 文档重构，改进双语支持
- 添加全面的故障排除指南

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

### 基础设施与文档

| 日期 | 描述 |
|------|-------------|
| 2026-03-13 | [文档与 CI 规范化](../zh/docs-ci-standardization) — CPU-safe CI，Pages 工作流修复 |
| 2026-03-10 | [GitHub Pages 增强](../zh/pages-enhancement) — SEO、导航、README 改进 |
| 2025-02-13 | [项目基础设施](../zh/project-infrastructure) — 许可证、编辑器配置、徽章 |

---

## 版本标签说明

| 类型 | 说明 |
|------|-------------|
| 🔴 **破坏性变更** | 需要迁移的破坏性变更 |
| 🟢 **新增** | 新功能 |
| 🔵 **修复** | Bug 修复 |
| 🟡 **变更** | 现有功能变更 |
| 🟣 **安全** | 安全相关变更 |

---

## 发布节奏

- **补丁版本** (x.y.Z): Bug 修复，按需每月发布
- **次要版本** (x.Y.z): 新功能，每季度发布
- **主版本** (X.y.z): 破坏性变更，每年发布

---

[← 返回首页](../) | [English Changelog](index) | [贡献指南](../CONTRIBUTING)
