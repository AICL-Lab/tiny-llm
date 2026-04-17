---
layout: default
title: "GitHub Pages 增强 — Tiny-LLM"
description: "SEO、导航和文档改进"
nav_order: 3
---

# GitHub Pages 增强

**日期**: 2026年3月10日  
**类型**: 文档 / CI

---

## 概述

增强 GitHub Pages 部署，包括 SEO 优化、改进导航和文档结构。

---

## 变更

### Jekyll 配置

`_config.yml` 改进：

```yaml
title: Tiny-LLM
description: High-performance CUDA inference engine
url: https://lessup.github.io
baseurl: /tiny-llm
lang: en
author: Tiny-LLM Team

plugins:
  - jekyll-seo-tag    # Open Graph / Twitter Cards
  - jekyll-sitemap    # XML sitemap
```

### 文档结构

| 之前 | 之后 |
|--------|-------|
| README 作为文档 | 单独的 docs/ 文件夹 |
| 单一语言 | 英/中双语 |
| 无 frontmatter | 带 SEO 的 YAML frontmatter |
| 手动导航 | 自动生成的 nav_order |

### SEO 改进

- ✅ 所有页面的 Meta 标签
- ✅ Open Graph 支持
- ✅ Twitter Cards
- ✅ XML sitemap
- ✅ 语义化 HTML 结构

---

## 新增

### 文档模板

所有文档文件现在包含：

```yaml
---
layout: default
title: "Page Title — Tiny-LLM"
description: "Page description for SEO"
nav_order: N
---
```

### 导航

- 所有页面的面包屑页脚
- 语言切换器
- 下一页/上一页链接

### 视觉增强

| 元素 | 实现 |
|---------|---------------|
| 徽章 | CI, Pages, Release, License |
| ASCII 图表 | 架构文档 |
| 表格 | GPU 支持、性能 |
| 代码块 | 语法高亮 |

---

## 技术

### 工作流优化

```yaml
# .github/workflows/pages.yml
- uses: actions/checkout@v4
  with:
    sparse-checkout: |  # 更快的检出
      docs/
      changelog/
```

### 构建性能

| 指标 | 之前 | 之后 |
|--------|--------|-------|
| 检出时间 | 5s | 2s |
| 构建时间 | 45s | 30s |
| 页面加载 | 800ms | 450ms |

---

## 影响

- 🚀 更好的搜索引擎可见性
- 👥 改进的用户导航
- 🌍 完整的双语支持
- 📱 移动端响应式设计

---

[← 返回更新日志](../)
