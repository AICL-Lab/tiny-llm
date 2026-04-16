# Changelog

All notable changes to the Tiny-LLM project are documented in the [changelog](changelog/) directory.

English | [中文](changelog/index.zh)

---

## Quick Links

| Version | Date | Type | English | 中文 |
|---------|------|------|---------|------|
| v2.0.1 | 2026-04-16 | Bug Fix | [Details](changelog/en/v2.0.1) | [详情](changelog/zh/v2.0.1) |
| v2.0.0 | 2026-03-09 | Major | [Details](changelog/en/v2.0.0) | [详情](changelog/zh/v2.0.0) |

---

## Release History

### [v2.0.1] - 2026-04-16

#### Fixed
- Critical `QuantizedWeight` scale dimension calculation error
- Removed unused code in attention kernel

**Details**: [English](changelog/en/v2.0.1) | [中文](changelog/zh/v2.0.1)

### [v2.0.0] - 2026-03-09

#### Changed (Breaking)
- KVCache API redesign with explicit `advanceSeqLen()`
- CMake modernization with target exports

**Details**: [English](changelog/en/v2.0.0) | [中文](changelog/zh/v2.0.0)

### Infrastructure Updates

| Date | Type | Description |
|------|------|-------------|
| 2026-03-13 | Docs/CI | [Documentation & CI Standardization](changelog/en/docs-ci-standardization) |
| 2026-03-10 | Docs/CI | [GitHub Pages Enhancement](changelog/en/pages-enhancement) |
| 2025-02-13 | Infra | [Project Infrastructure](changelog/en/project-infrastructure) |

---

## Format

This project adheres to:
- [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
- [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

[Documentation](https://lessup.github.io/tiny-llm/) | [API Reference](https://lessup.github.io/tiny-llm/docs/en/API) | [Contributing](CONTRIBUTING.md)
