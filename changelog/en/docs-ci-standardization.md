---
layout: default
title: "Documentation & CI Standardization"
description: "Documentation separation and CPU-safe CI"
nav_order: 4
---

# Documentation & CI Standardization

**Date**: March 13, 2026  
**Type**: Documentation / CI

---

## Documentation Restructuring

### README vs Index Separation

| File | Before | After |
|------|--------|-------|
| `README.md` | Full project docs | Repository entry only |
| `README.zh-CN.md` | Full project docs | Repository entry only |
| `index.md` | Duplicated README | Documentation homepage |

### Documentation Homepage Features

- Project positioning statement
- "Who is it for" section
- "Where to start" recommendations
- Document index table

---

## CI Improvements

### CPU-safe Configuration

**Problem**: CI depended on CUDA containers but GitHub runners lack GPUs.

**Solution**: 
- Removed CUDA build from CI
- Kept format check using `jidicula/clang-format-action@v4.13.0`
- Excluded build directories from format check

### Pages Workflow Fix

- Expanded branch triggers from `main` to `master, main`
- Better sparse checkout configuration

---

## Verification

- ✅ README and index.md responsibilities separated
- ✅ Home page navigation links verified
- ✅ Pages workflow covers all branches

---

[← Back to Changelog](../)
