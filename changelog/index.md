---
layout: default
title: "Changelog — Tiny-LLM"
description: "Tiny-LLM version history and release notes"
---

# Changelog

Version history for the Tiny-LLM project.

---

## Language Selection

- **English** (this page)
- [简体中文](index.zh)

---

## Recent Releases

| Date | Version | Type | Summary |
|------|---------|------|---------|
| April 16, 2026 | [v2.0.1](en/v2.0.1) | Bug Fix | Scale dimension fix, code cleanup |
| March 9, 2026 | [v2.0.0](en/v2.0.0) | Core | KVCache API redesign, CMake modernization |

## Documentation & CI Updates

| Date | Version | Type | Summary |
|------|---------|------|---------|
| March 13, 2026 | — | Docs/CI | Documentation separation, CPU-safe CI |
| March 10, 2026 | — | Docs/CI | GitHub Pages enhancement, SEO |
| February 13, 2025 | — | Infra | LICENSE, editorconfig, badges |

---

## Version Legend

| Type | Description |
|------|-------------|
| **Core** | Core functionality changes |
| **Bug Fix** | Bug fixes |
| **Docs** | Documentation changes |
| **CI** | CI/CD changes |
| **Infra** | Infrastructure changes |

---

## Full Changelog History

### v2.0.1 (2026-04-16)

**Fixed**:
- Critical `QuantizedWeight` scale dimension calculation error
- Removed unused code in attention kernel

[View Details →](en/v2.0.1)

### v2.0.0 (2026-03-09)

**Changed**:
- **Breaking**: KVCache API redesign with `advanceSeqLen()`
- CMake modernization

[View Details →](en/v2.0.0)

### Earlier Updates

- [GitHub Pages Enhancement](en/pages-enhancement) (2026-03-10)
- [Documentation & CI Standardization](en/docs-ci-standardization) (2026-03-13)
- [Project Infrastructure](en/project-infrastructure) (2025-02-13)

---

[← Home](../)
