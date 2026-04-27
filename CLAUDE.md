# CLAUDE.md

This file defines the preferred Claude/agent operating style for Tiny-LLM.

## Mission

Help finish the repository cleanly:

- tighten OpenSpec compliance
- reduce document and workflow drift
- keep the CUDA/C++ core reliable
- make the public project surface sharper
- avoid speculative expansion

## Default workflow

1. Read the active OpenSpec change and current specs first.
2. If the task is non-trivial and no change exists, propose one before broad edits.
3. Implement from `tasks.md` in dependency order.
4. Run a review pass after major workstreams.
5. Archive the change when code, docs, and specs agree.

## Editing posture

- Prefer precise rewrites over layering more generic text.
- Prefer deleting stale material over “keeping both”.
- Keep CI non-mutating.
- Keep user-facing claims conservative and verifiable.
- Respect existing user changes in the working tree; reconcile before overwriting.

## Tooling posture

- Use `gh` for repository metadata work.
- Treat `clangd` + `compile_commands.json` as the canonical LSP baseline.
- Keep MCP/plugin usage optional and narrow.
- Prefer long single-session execution over `/fleet`.

## Technical reminders

- C++17 + CUDA C++17
- `Result<T>` instead of control-flow exceptions
- RAII for resource ownership
- `clang-format-18`
- `nvcc` required for real configure/build/test passes

## Core architecture

| Component | Responsibility |
|-----------|----------------|
| `Result<T>` | No-exception error propagation |
| `ModelConfig` | Model hyperparameters (vocab_size, hidden_dim, etc.) |
| `QuantizedWeight` | INT8 weights with per-group scales |
| `TransformerLayer` | W8A16 quantized attention + FFN |
| `KVCacheManager` | Pre-allocated cache slots for sequences |
| `InferenceEngine` | Public API: load(), generate() |

## Validation commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
```

## Local overrides

Personal overrides may live in `CLAUDE.local.md`, but that file is local-only and must stay untracked.
