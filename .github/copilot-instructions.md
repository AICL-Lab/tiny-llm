# Copilot instructions for Tiny-LLM

## Project mode

Tiny-LLM is in a final hardening phase. Prefer cleanup, alignment, and reliability work over new roadmap expansion.

## Required workflow

- Use OpenSpec for non-trivial work.
- Read the active change in `openspec/changes/<name>/` before editing.
- Treat `openspec/specs/` as the current behavior contract.
- Keep changes scoped to the active task instead of mixing unrelated cleanup.

## Repository-specific rules

- Do not reintroduce legacy `/specs` guidance.
- Prefer deleting or merging stale docs over adding parallel explanations.
- Keep README, Pages, changelog surfaces, and GitHub metadata aligned.
- Do not assume CI will auto-format or auto-commit fixes.
- Use `gh` for repository metadata updates.

## Engineering constraints

- C++17 and CUDA C++17
- `Result<T>` for fallible operations
- RAII for ownership
- `clang-format-18`
- `clangd` + `compile_commands.json` as the default language-intelligence baseline

## Validation commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
```

`nvcc` must be available for configure/build/test to succeed.
