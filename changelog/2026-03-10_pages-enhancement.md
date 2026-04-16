---
layout: default
title: "GitHub Pages 完善与优化 (2026-03-10)"
description: "SEO 配置、导航完善、文档结构优化、sparse checkout"
nav_order: 3
---

# 2026-03-10 — GitHub Pages 完善与优化

**类型**: 文档 / CI

---

## Jekyll 配置增强

**文件**: `_config.yml`

| 配置项 | 变更 |
|--------|------|
| SEO 元数据 | 添加 `url`、`baseurl`、`lang`、`author` |
| SEO 插件 | 添加 `jekyll-seo-tag`（自动注入 Open Graph / Twitter Card） |
| Markdown | 明确 `kramdown` + GFM 输入 + `rouge` 语法高亮 |
| 默认布局 | 全局 `layout: default`，changelog 目录也应用默认布局 |
| 排除列表 | 排除源代码（`include`/`kernels`/`src`/`tests`）、构建配置等 |

---

## Pages 工作流优化

**文件**: `.github/workflows/pages.yml`

| 优化项 | 描述 |
|--------|------|
| Sparse checkout | 仅检出文档相关文件，跳过 C++/CUDA 源代码 |
| Cone mode 修正 | 设置 `sparse-checkout-cone-mode: false` 修复文件级匹配 |
| 路径触发 | 从通配 `*.md` 改为显式列出文档路径 + `changelog/**` |
| Job 名称 | 添加 `Build Pages` / `Deploy Pages` |

---

## 文档页面增强

### API 文档 (`docs/API.md`)
- 添加 YAML frontmatter（title + description）
- 添加导航页脚（返回首页 / 更新日志 / 贡献指南）

### 贡献指南 (`CONTRIBUTING.md`)
- 添加 YAML frontmatter
- 添加导航页脚

### 首页 (`index.md`)
- 添加 SEO frontmatter
- 添加"最近更新" section
- 重构文档链接为中文
- 添加更新日志和贡献指南入口

---

## README 增强

### 英文版 (`README.md`)
- 添加 CI/Pages 徽章
- 添加 CMake 徽章
- 添加 ASCII 架构图
- 添加 GPU 架构支持表（Volta → Hopper）
- 添加测试详情表
- 扩展项目结构说明

### 中文版 (`README.zh-CN.md`)
- 添加 CI/Pages 徽章
- 添加架构图
- 添加性能优化表（6 项优化技术）
- 添加 GPU 架构支持表
- 添加技术栈表
- 扩展项目结构说明

---

## 其他变更

- `.gitignore` 添加 `.cache/`（排除 clangd 缓存）
- `pages.yml` sparse-checkout 路径规范化（`docs/`、`changelog/`）

---

## 文件变更摘要

| 文件 | 变更类型 |
|------|----------|
| `_config.yml` | SEO、插件、布局配置 |
| `.github/workflows/pages.yml` | Sparse checkout、路径触发 |
| `docs/API.md` | Frontmatter、导航 |
| `CONTRIBUTING.md` | Frontmatter、导航 |
| `index.md` | SEO、"最近更新" section |
| `README.md` | 徽章、架构图、GPU 表 |
| `README.zh-CN.md` | 徽章、架构图、性能表 |
| `.gitignore` | 添加 `.cache/` |

---

[← 返回更新日志](index)
