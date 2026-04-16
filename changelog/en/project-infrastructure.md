---
layout: default
title: "Project Infrastructure"
description: "Initial project setup with LICENSE and editor configuration"
nav_order: 5
---

# Project Infrastructure

**Date**: February 13, 2025  
**Type**: Infrastructure

---

## Project Init

Established core project files and configuration.

### New Files

| File | Purpose |
|------|---------|
| `LICENSE` | MIT License |
| `.editorconfig` | Editor configuration (encoding, indentation, line endings) |

### README Badges

- License: MIT
- CUDA: 11.0+
- C++: 17
- CMake: 3.18+

---

## Editor Configuration

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

[← Back to Changelog](../)
