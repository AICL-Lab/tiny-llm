---
title: "v2.0.0 — Major Refactoring (2026-03-09)"
---

# Major Refactoring - v2.0.0

Date: 2026-03-09

## Critical Bug Fixes

### KVCache appendKV: layer-order-dependent write position logic
- `appendKV()` had fragile write position logic that depended on layer 0 being called first
- Layer 0 updated `current_len`, then other layers tried to compensate by subtracting `num_tokens` — broke if layers were called in any other order
- Negative `write_pos` possible if layer 0 hadn't been called yet
- **Fix**: appendKV now always writes at `current_len` regardless of layer index (stateless per-layer)
- Added explicit `advanceSeqLen()` method to be called ONCE after all layers have appended
- Clean separation: append is stateless per-layer, length update is explicit per-step

## Build System

### CMakeLists.txt modernization
- Project `VERSION 2.0.0`, `DESCRIPTION`
- CUDA arch auto-detect (`native` on CMake 3.24+, fallback to common archs)
- Replaced global `include_directories()` with proper `target_include_directories()` (PUBLIC/PRIVATE)
- Added `tiny_llm::tiny_llm` ALIAS target
- Excluded `main.cpp` from library sources (was being linked into both library and demo executable)
- Added compiler warnings (`-Wall -Wextra` / `/W4`)
- `gtest_force_shared_crt` for MSVC
- `CMAKE_EXPORT_COMPILE_COMMANDS` for IDE support
- Removed unused RapidCheck dependency
- Wrapped tests in `BUILD_TESTS` option

### Files Modified
- `src/kv_cache.cpp` — stateless appendKV, new advanceSeqLen
- `include/tiny_llm/kv_cache.h` — advanceSeqLen declaration
- `CMakeLists.txt` — full modernization
