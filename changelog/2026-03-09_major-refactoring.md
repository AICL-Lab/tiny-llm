---
layout: default
title: "v2.0.0 — 重大重构 (2026-03-09)"
description: "KVCache 层序依赖修复、CMake 现代化、CI 工作流"
nav_order: 4
---

# v2.0.0 — 重大重构

**发布日期**: 2026-03-09  
**类型**: 核心修复 / 构建 / CI

---

## Critical Bug 修复

### KVCache appendKV 层序依赖问题

**严重程度**: Critical  
**文件**: `src/kv_cache.cpp`, `include/tiny_llm/kv_cache.h`

**问题描述**:
`appendKV()` 的写入位置逻辑存在脆弱的层序依赖：
- Layer 0 更新 `current_len`，其他层尝试通过减去 `num_tokens` 来补偿
- 如果层调用顺序不同，逻辑会崩溃
- 可能产生负的 `write_pos`

**修复方案**:
1. `appendKV` 改为无状态写入
   - 所有层都写入到 `current_len` 位置
   - 不再依赖层索引
2. 新增 `advanceSeqLen()` 方法
   - 在所有层完成 append 后显式调用一次
   - 清晰的职责分离：append 是每层的无状态操作，长度更新是每步的显式操作

```cpp
// 修复后的使用模式
for (auto &layer : layers_) {
  layer->forward(hidden_states, *kv_cache_, seq_id, position, stream_);
}
kv_cache_->advanceSeqLen(seq_id, 1);  // 显式更新
```

---

## 构建系统现代化

**文件**: `CMakeLists.txt`

| 变更项 | 描述 |
|--------|------|
| 项目版本 | 升级至 `VERSION 2.0.0` |
| 项目描述 | 添加 `DESCRIPTION` |
| CUDA 架构 | 自动检测（CMake 3.24+ 使用 `native`，回退到常见架构） |
| Include 路径 | 使用 `target_include_directories()` 替代全局 `include_directories()` |
| Alias Target | 添加 `tiny_llm::tiny_llm` |
| 源文件排除 | 从库源文件中排除 `main.cpp` |
| 编译警告 | 添加 `-Wall -Wextra`（GCC/Clang）、`/W4`（MSVC） |
| IDE 支持 | 生成 `compile_commands.json` |
| 测试构建 | 包装在 `BUILD_TESTS` 选项中 |

---

## CI 工作流

**文件**: `.github/workflows/ci.yml`

- 标准化触发：`push`、`pull_request`、`workflow_dispatch`
- CUDA 容器构建验证
- `clang-format` 格式检查

---

## 文件变更摘要

| 文件 | 变更类型 |
|------|----------|
| `src/kv_cache.cpp` | Bug Fix: 无状态 appendKV |
| `include/tiny_llm/kv_cache.h` | 新增 `advanceSeqLen()` |
| `CMakeLists.txt` | 全面现代化 |
| `.github/workflows/ci.yml` | 新增 CI 工作流 |

---

[← 返回更新日志](index)
