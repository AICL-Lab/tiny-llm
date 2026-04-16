---
layout: default
title: "项目基础设施"
description: "初始项目设置，包含 LICENSE 和编辑器配置"
nav_order: 5
---

# 项目基础设施

**日期**: 2025年2月13日  
**类型**: 基础设施

---

## 项目初始化

建立核心项目文件和配置。

### 新增文件

| 文件 | 用途 |
|------|------|
| `LICENSE` | MIT 许可证 |
| `.editorconfig` | 编辑器配置（编码、缩进、换行符） |

### README 徽章

- License: MIT
- CUDA: 11.0+
- C++: 17
- CMake: 3.18+

---

## 编辑器配置

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

[*.{yml,yaml,json}]
indent_size = 2
```

---

[← 返回更新日志](../)
