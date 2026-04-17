---
layout: default
title: "Documentation & CI Standardization — Tiny-LLM"
description: "Documentation separation and CPU-safe CI configuration"
nav_order: 4
---

# Documentation & CI Standardization

**Date**: March 13, 2026  
**Type**: Documentation / CI

---

## Overview

Separated documentation responsibilities and modernized CI configuration for CPU-only GitHub runners.

---

## Changed

### Documentation Structure

| File | Before | After |
|------|--------|-------|
| `README.md` | Full documentation | Repository entry point only |
| `index.md` | Duplicate of README | Documentation homepage |
| `docs/en/`, `docs/zh/` | Did not exist | Full documentation in both languages |
| `changelog/` | Single file | Structured directory with EN/ZH |

#### Responsibility Separation

| Document | Purpose |
|----------|---------|
| README | Project overview, badges, quick links |
| index.md | Documentation landing with navigation |
| docs/en/, docs/zh/ | Complete API and architecture docs |
| changelog/ | Version history and release notes |

### CI/CD Modernization

#### Problem
GitHub Actions runners don't have GPUs, making CUDA builds fail.

#### Solution
```yaml
# ci.yml - Simplified for CPU-only runners
jobs:
  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: jidicula/clang-format-action@v4.13.0
        with:
          clang-format-version: '17'
          check-path: '.'
          
  # CUDA builds removed (run locally or on GPU runners)
```

#### Pages Workflow

```yaml
# pages.yml - Enhanced for broader branch support
on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

# Sparse checkout for faster deployment
- uses: actions/checkout@v4
  with:
    sparse-checkout: |
      docs/
      changelog/
```

---

## Added

### Format Validation

```yaml
- name: Check C++ formatting
  uses: jidicula/clang-format-action@v4.13.0
  with:
    clang-format-version: '17'
    exclude-regex: '(build/|third_party/)'
```

### Ignore Patterns

| Pattern | Reason |
|---------|--------|
| `build/**` | CMake build directory |
| `**/CMakeFiles/**` | CMake intermediates |
| `**/*.cmake` | Generated files |

---

## Removed

### From CI

- ❌ CUDA compilation steps
- ❌ GPU-dependent tests
- ❌ `nvidia/cuda` containers

### From README

- ❌ Detailed API documentation
- ❌ Architecture diagrams
- ❌ Long usage examples

---

## Migration

### For Contributors

**Before**:
```bash
# Wait for CI GPU runners
```

**After**:
```bash
# Local validation before push
./scripts/format-check.sh
mkdir build && cd build
cmake .. && make -j$(nproc)
ctest
```

---

## Impact

| Aspect | Before | After |
|--------|--------|-------|
| CI reliability | Unreliable | ✅ Stable |
| CI runtime | 10+ min | ✅ 30s |
| PR friction | High | ✅ Low |
| Doc clarity | Mixed | ✅ Focused |

---

[← Back to Changelog](../)
