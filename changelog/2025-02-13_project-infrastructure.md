---
layout: default
title: "项目基础设施 (2025-02-13)"
description: "LICENSE、editorconfig、badges"
nav_order: 5
---

# 2025-02-13 — 项目基础设施

**类型**: 基础设施

---

## 新增文件

| 文件 | 描述 |
|------|------|
| `LICENSE` | MIT 许可证 |
| `.editorconfig` | 统一编辑器配置（编码、缩进、换行符） |

---

## README 增强

添加标准化徽章：

- License: MIT
- CUDA: 11.0+
- C++: 17
- CMake: 3.18+

---

## .editorconfig 配置

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

[← 返回更新日志](index)
