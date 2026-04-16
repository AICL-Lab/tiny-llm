---
layout: default
title: "文档与 CI 规范化"
description: "文档入口分离和 CPU-safe CI"
nav_order: 4
---

# 文档与 CI 规范化

**日期**: 2026年3月13日  
**类型**: 文档 / CI

---

## 文档架构调整

### README 与首页职责分离

| 文件 | 调整前 | 调整后 |
|------|--------|------|
| `README.md` | 完整项目文档 | 仅仓库入口 |
| `README.zh-CN.md` | 完整项目文档 | 仅仓库入口 |
| `index.md` | 重复的 README | 文档首页 |

### 文档首页功能

- 项目定位说明
- "适合谁" 章节
- "从哪里开始" 推荐
- 文档索引表

---

## CI 改进

### CPU-safe 配置

**问题**: CI 依赖 CUDA 容器，但 GitHub runner 没有 GPU。

**解决方案**: 
- 从 CI 中移除 CUDA 构建
- 保留格式检查，使用 `jidicula/clang-format-action@v4.13.0`
- 从格式检查中排除构建目录

### Pages 工作流修复

- 分支触发器从 `main` 扩展到 `master, main`
- 更好的 sparse checkout 配置

---

## 验证

- ✅ README 和 index.md 职责已分离
- ✅ 首页导航链接已验证
- ✅ Pages 工作流覆盖所有分支

---

[← 返回更新日志](../)
