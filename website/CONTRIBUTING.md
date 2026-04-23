---
layout: page
title: "Contributing — Tiny-LLM"
description: "How to contribute to Tiny-LLM project"
nav_order: 50
---

# Contributing to Tiny-LLM

Tiny-LLM is in a final hardening phase. Contributions are welcome, but the repository now favors **focused, OpenSpec-driven improvements** over broad feature expansion.

---

## Required workflow

1. Use `/opsx:explore` when the scope or requirements are still fuzzy.
2. Use `/opsx:propose <change-name>` for non-trivial work so the change is captured in `openspec/`.
3. Use `/opsx:apply <change-name>` to implement task-by-task.
4. Run a review pass before concluding major workstreams.
5. Use `/opsx:archive <change-name>` only after specs, code, and docs are aligned.

> Repository-specific workflow and architecture constraints live in [AGENTS.md](https://github.com/LessUp/tiny-llm/blob/master/AGENTS.md).

---

## Build and validate

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
```

---

## Contribution principles

- Keep branches short-lived and avoid local/cloud drift.
- Prefer deleting or merging duplicated docs over preserving parallel guidance.
- Keep public claims aligned with the actual implementation and validated documentation.
- Treat CI as validation, not as a system that rewrites tracked files for contributors.

---

## Where to look next

- [Repository guide](https://github.com/LessUp/tiny-llm/blob/master/AGENTS.md)
- [Current specs](https://github.com/LessUp/tiny-llm/tree/master/openspec/specs)
- [Open changes](https://github.com/LessUp/tiny-llm/tree/master/openspec/changes)

---

[← Home](/) | [Docs](/docs/en/) | [GitHub](https://github.com/LessUp/tiny-llm)
