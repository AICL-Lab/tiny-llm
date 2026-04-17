---
layout: default
title: "项目基础设施 — Tiny-LLM"
description: "初始项目设置，包含许可证和编辑器配置"
nav_order: 5
---

# 项目基础设施

**日期**: 2025年2月13日  
**类型**: 基础设施设置

---

## 概述

建立项目基础结构和配置文件。

---

## 新增

### 法律和治理

| 文件 | 用途 |
|------|---------|
| `LICENSE` | MIT 许可证 — 宽松的开源许可 |
| `CODE_OF_CONDUCT.md` | 社区行为准则 |
| `CONTRIBUTING.md` | 贡献指南 |
| `SECURITY.md` | 安全政策 |

### 开发环境

| 文件 | 配置 |
|------|---------------|
| `.editorconfig` | UTF-8, LF, C++ 4 空格缩进 |
| `.gitignore` | CMake 构建产物, IDE 文件 |
| `.gitattributes` | 换行符规范化 |

#### EditorConfig 设置

```ini
[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4

[*.{h,hpp,c,cpp,cu,cuh}]
indent_size = 4

[*.{yml,yaml,json,md}]
indent_size = 2
```

### 文档

- 带徽章的项目 README
- 初始项目描述
- 基础构建说明

### 徽章

```markdown
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)
```

---

## 影响

这些文件建立了：
- ✅ 清晰的许可条款
- ✅ 一致的代码风格
- ✅ 协作开发标准
- ✅ 专业的项目形象

---

[← 返回更新日志](../)
