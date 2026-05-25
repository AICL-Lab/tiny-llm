# Copilot instructions for Tiny-LLM

## Project mode

Tiny-LLM is in a final hardening phase. Prefer cleanup, consolidation, and reliability work over roadmap expansion.

## Repository priorities

- Keep the repository free of AI-specific governance scaffolding and duplicate instruction surfaces.
- Keep README, GitHub Pages, and the root `CHANGELOG.md` aligned with the actual project.
- Prefer deleting or rewriting stale docs over preserving parallel explanations.
- Do not assume CI will auto-format or auto-commit fixes.

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
