---
layout: default
title: "文档与 CI 规范化 (2026-03-13)"
description: "文档入口分离、Pages workflow 分支触发修正、CI 调整为 CPU-safe"
nav_order: 2
---

# 2026-03-13 — 文档与 CI 规范化

**类型**: 文档 / CI

---

## 文档架构调整

### README 与首页职责分离

**背景**: 此前 `README.md`、`README.zh-CN.md` 与 `index.md` 都承担了较完整的项目说明，仓库入口与文档入口重复明显。

**变更**:

| 文件 | 调整前 | 调整后 |
|------|--------|--------|
| `README.md` | 完整项目说明 | 仓库入口，仅保留定位、构建命令、文档链接 |
| `README.zh-CN.md` | 完整项目说明 | 仓库入口，仅保留定位、构建命令、文档链接 |
| `index.md` | 与 README 重复 | 文档首页，提供导读和阅读路径 |

**新增内容** (`index.md`):
- 项目定位说明
- "适合谁" 章节
- "从哪里开始" 推荐
- 推荐阅读路径
- 核心文档索引表

---

## CI 工作流调整

### CPU-safe CI 配置

**文件**: `.github/workflows/ci.yml`

**背景**: 该仓库原先的 CI 依赖 `nvidia/cuda` 容器，但 GitHub Hosted Runner 不提供可用 GPU，导致工作流长期处于失败状态。

**变更**:
- 移除 CUDA 构建检查
- 保留 `Format Check` job（clang-format 校验）
- 使用 `jidicula/clang-format-action@v4.13.0`
- 排除 `build`、`third_party`、`external`、`vendor` 目录

### Pages workflow 分支触发修正

**文件**: `.github/workflows/pages.yml`

**问题**: Pages workflow 只监听 `main` 分支，而仓库当前分支为 `master`，导致文档修改无法自动触发部署。

**修复**: 触发分支从 `main` 扩展为 `master, main`。

---

## 验证结果

- ✅ README 与 index.md 职责已分离
- ✅ 首页到各文档链接正确
- ✅ Pages workflow 已覆盖实际使用分支

---

## 文件变更摘要

| 文件 | 变更类型 |
|------|----------|
| `README.md` | 精简为仓库入口 |
| `README.zh-CN.md` | 精简为仓库入口 |
| `index.md` | 改写为文档首页 |
| `.github/workflows/ci.yml` | CPU-safe 调整 |
| `.github/workflows/pages.yml` | 分支触发修正 |

---

[← 返回更新日志](index)
