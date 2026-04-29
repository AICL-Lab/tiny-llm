---
layout: default
title: "Changelog — Tiny-LLM"
description: "Tiny-LLM version history and release notes"
nav_order: 100
---

# Changelog

Tiny-LLM keeps a short, release-oriented changelog. This surface is reserved for meaningful external milestones.

---

## Language Selection

- **English** (this page)
- [简体中文](index.zh)

---

## Releases

### [2.0.2] — 2026-04-27

#### Added
- **Quantization utilities**: New `quantization.h`/`quantization.cpp` with F32→F16, Q4_0, Q8_0, W8A16 utilities
- **CLI enhancements**: `--help`, `--version`, `--info` options for better usability

#### Changed
- **KVCacheManager**: Factory method `create()` for consistent `Result<T>` error handling
- **Code quality**: Added `noexcept` to simple accessors, fixed clang-format-18 violations

#### Infrastructure
- CI simplification with `Jimver/cuda-toolkit` action
- Enhanced `.clangd` LSP configuration

**Details**: [English](en/v2.0.2) | [中文](zh/v2.0.2)

---

### [2.0.1] — 2026-04-16

#### Fixed
- **CRITICAL**: `QuantizedWeight` scale dimension calculation error in test utilities
- Removed unused code in attention kernel (`q_reg` array loading)

**Details**: [English](en/v2.0.1) | [中文](zh/v2.0.1)

---

### [2.0.0] — 2026-03-09

#### Changed  ⚠️ BREAKING
- **API Redesign**: KVCache `appendKV()` is now stateless with explicit `advanceSeqLen()`
- CMake modernization with target exports and architecture auto-detection

#### Added
- CI workflow with automated format checking
- `tiny_llm::tiny_llm` CMake alias target

**Migration Guide**: Update any direct KVCache usage to call `advanceSeqLen()` after all layers.

**Details**: [English](en/v2.0.0) | [中文](zh/v2.0.0)

---

## Notes

- Infrastructure-only cleanup is intentionally excluded from this public changelog.
- For the current project story and onboarding flow, start at [Home](../) or [Documentation](../docs/).

---

[← Home](../) | [中文更新日志](index.zh) | [Contributing](../CONTRIBUTING)
