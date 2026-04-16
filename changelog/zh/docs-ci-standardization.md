---
layout: default
title: "文档与 CI 规范化 — Tiny-LLM"
description: "文档分离和 CPU-safe CI 配置"
nav_order: 4
---

# 文档与 CI 规范化

**日期**: 2026年3月13日  
**类型**: 文档 / CI

---

## 概述

分离文档职责，并为仅 CPU 的 GitHub runners 现代化 CI 配置。

---

## 变更

### 文档结构

| 文件 | 之前 | 之后 |
|------|--------|-------|
| `README.md` | 完整文档 | 仅作为仓库入口 |
| `index.md` | README 的副本 | 文档首页 |
| `docs/en/`, `docs/zh/` | 不存在 | 双语完整文档 |
| `changelog/` | 单一文件 | 带 EN/ZH 的结构化目录 |

#### 职责分离

| 文档 | 用途 |
|----------|---------|
| README | 项目概览、徽章、快速链接 |
| index.md | 带导航的文档落地页 |
| docs/en/, docs/zh/ | 完整的 API 和架构文档 |
| changelog/ | 版本历史和发布说明 |

### CI/CD 现代化

#### 问题
GitHub Actions runners 没有 GPU，导致 CUDA 构建失败。

#### 解决方案
```yaml
# ci.yml - 为仅 CPU runners 简化
jobs:
  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: jidicula/clang-format-action@v4.13.0
        with:
          clang-format-version: '17'
          check-path: '.'
          
  # CUDA 构建移除（在本地或 GPU runners 上运行）
```

#### Pages 工作流

```yaml
# pages.yml - 增强分支支持
on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

# 稀疏检出以加快部署
- uses: actions/checkout@v4
  with:
    sparse-checkout: |
      docs/
      changelog/
```

---

## 新增

### 格式验证

```yaml
- name: Check C++ formatting
  uses: jidicula/clang-format-action@v4.13.0
  with:
    clang-format-version: '17'
    exclude-regex: '(build/|third_party/)'
```

### 忽略模式

| 模式 | 原因 |
|---------|--------|
| `build/**` | CMake 构建目录 |
| `**/CMakeFiles/**` | CMake 中间文件 |
| `**/*.cmake` | 生成的文件 |

---

## 移除

### 从 CI

- ❌ CUDA 编译步骤
- ❌ GPU 依赖的测试
- ❌ `nvidia/cuda` 容器

### 从 README

- ❌ 详细的 API 文档
- ❌ 架构图表
- ❌ 长示例代码

---

## 迁移

### 对于贡献者

**之前**:
```bash
# 等待 CI GPU runners
```

**之后**:
```bash
# 推送前本地验证
./scripts/format-check.sh
mkdir build && cd build
cmake .. && make -j$(nproc)
ctest
```

---

## 影响

| 方面 | 之前 | 之后 |
|--------|--------|-------|
| CI 可靠性 | 不可靠 | ✅ 稳定 |
| CI 运行时间 | 10+ 分钟 | ✅ 30秒 |
| PR 摩擦 | 高 | ✅ 低 |
| 文档清晰度 | 混合 | ✅ 专注 |

---

[← 返回更新日志](../)
