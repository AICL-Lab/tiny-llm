---
layout: default
title: "更新日志 — Tiny-LLM"
description: "Tiny-LLM 版本历史和发布说明"
---

# 更新日志

Tiny-LLM 项目版本历史。

---

## 语言选择

- **简体中文** (本页)
- [English](index)

---

## 最近发布

| 日期 | 版本 | 类型 | 摘要 |
|------|---------|------|------|
| 2026-04-16 | [v2.0.1](zh/v2.0.1) | Bug Fix | 尺度维度修复，代码清理 |
| 2026-03-09 | [v2.0.0](zh/v2.0.0) | Core | KVCache API 重新设计，CMake 现代化 |

## 文档与 CI 更新

| 日期 | 版本 | 类型 | 摘要 |
|------|---------|------|------|
| 2026-03-13 | — | Docs/CI | 文档入口分离，CPU-safe CI |
| 2026-03-10 | — | Docs/CI | GitHub Pages 完善，SEO |
| 2025-02-13 | — | Infra | LICENSE，editorconfig，徽章 |

---

## 版本类型说明

| 类型 | 说明 |
|------|------|
| **Core** | 核心功能变更 |
| **Bug Fix** | Bug 修复 |
| **Docs** | 文档变更 |
| **CI** | CI/CD 变更 |
| **Infra** | 基础设施变更 |

---

## 完整更新历史

### v2.0.1 (2026-04-16)

**修复**:
- Critical `QuantizedWeight` 尺度维度计算错误
- 移除注意力 kernel 中未使用的代码

[查看详情 →](zh/v2.0.1)

### v2.0.0 (2026-03-09)

**变更**:
- **破坏性变更**: KVCache API 重新设计，新增 `advanceSeqLen()`
- CMake 现代化

[查看详情 →](zh/v2.0.0)

### 早期更新

- [GitHub Pages 完善](zh/pages-enhancement) (2026-03-10)
- [文档与 CI 规范化](zh/docs-ci-standardization) (2026-03-13)
- [项目基础设施](zh/project-infrastructure) (2025-02-13)

---

[← 返回首页](../)
