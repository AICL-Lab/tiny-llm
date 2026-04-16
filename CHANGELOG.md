# Changelog

All notable changes to the Tiny-LLM project are documented in the [changelog](changelog/) directory.

## Quick Links

- [English Changelog](changelog/)
- [中文更新日志](changelog/index.zh)

## Recent Releases

### [v2.0.1] - 2026-04-16

**Fixed**:
- Critical `QuantizedWeight` scale dimension calculation error
- Removed unused code in attention kernel

**Full details**: [English](changelog/en/v2.0.1) | [中文](changelog/zh/v2.0.1)

### [v2.0.0] - 2026-03-09

**Changed** (Breaking):
- KVCache API redesign: added `advanceSeqLen()` method
- CMake modernization with alias targets

**Full details**: [English](changelog/en/v2.0.0) | [中文](changelog/zh/v2.0.0)

---

## Format

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and uses [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## Types of Changes

| Type | Description |
|------|-------------|
| `feat` | New features |
| `fix` | Bug fixes |
| `docs` | Documentation only changes |
| `perf` | Performance improvements |
| `refactor` | Code refactoring |
| `test` | Adding or correcting tests |
| `ci` | CI/CD changes |

---

**Languages**: [English](changelog/) | [中文](changelog/index.zh)
