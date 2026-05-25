# Contributing to Tiny-LLM

Tiny-LLM is in a hardening and cleanup phase. Prefer focused, reviewable changes that make the repository smaller, clearer, and more accurate.

## Working rules

1. **Keep one source of truth per topic.** If two documents explain the same thing, merge or delete one.
2. **Do not reintroduce workflow scaffolding.** Avoid adding repository-level agent command packs, memory dumps, spec trees, or parallel instruction documents.
3. **Keep release history in one place.** The root `CHANGELOG.md` is the only tracked changelog.
4. **CI validates, contributors fix.** Do not rely on automation to rewrite tracked files for you.

## Typical workflow

1. Read the current code and the directly relevant docs.
2. Make one coherent change at a time.
3. Update user-facing docs when behavior, supported workflows, or public links change.
4. Run the validation commands before concluding your work.

## Build and validation

Tiny-LLM requires a working CUDA toolchain (`nvcc` on `PATH` or an equivalent configured CUDA installation).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
```

For formatting checks:

```bash
find . -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
  ! -path './build/*' | xargs clang-format-18 --dry-run --Werror
```

## Documentation rules

- Keep README, GitHub Pages, badges, clone links, and issue links aligned with the same repository identity.
- Keep public claims conservative and verifiable.
- If a page becomes stale, duplicated, or lower-signal than the code itself, rewrite it or remove it.
- The documentation site should document the engine, not mirror release history or process bureaucracy.

## Project-specific constraints

- Language/toolchain: **C++17 + CUDA C++17**
- Error model: prefer **`Result<T>`** for fallible operations
- Ownership model: prefer **RAII** over raw lifetime management
- Formatting: **`clang-format-18`**
- Language intelligence baseline: **`clangd` + `compile_commands.json`**
