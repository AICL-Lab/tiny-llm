---
layout: default
title: "Changelog — Tiny-LLM"
description: "Tiny-LLM version history and release notes"
nav_order: 100
---

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Language Selection

- **English** (this page)
- [简体中文](index.zh)

---

## Release History

### [Unreleased]

- Documentation restructuring with improved bilingual support
- Added comprehensive troubleshooting guides

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

### Infrastructure & Documentation

| Date | Description |
|------|-------------|
| 2026-03-13 | [Documentation & CI Standardization](en/docs-ci-standardization) — CPU-safe CI, Pages workflow fixes |
| 2026-03-10 | [GitHub Pages Enhancement](en/pages-enhancement) — SEO, navigation, README improvements |
| 2025-02-13 | [Project Infrastructure](en/project-infrastructure) — LICENSE, editorconfig, badges |

---

## Version Legend

| Type | Description |
|------|-------------|
| 🔴 **Breaking** | Breaking changes requiring migration |
| 🟢 **Added** | New features |
| 🔵 **Fixed** | Bug fixes |
| 🟡 **Changed** | Changes to existing functionality |
| 🟣 **Security** | Security-related changes |

---

## Release Schedule

- **Patch releases** (x.y.Z): Bug fixes, monthly as needed
- **Minor releases** (x.Y.z): New features, quarterly
- **Major releases** (X.y.z): Breaking changes, annually

---

[← Home](../) | [中文更新日志](index.zh) | [Contributing](../CONTRIBUTING)
