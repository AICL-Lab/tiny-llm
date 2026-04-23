# Tiny-LLM Agent Guide

Tiny-LLM is in a **final hardening / archive-readiness** phase. Treat this repository as a CUDA/C++ engine that should become **smaller, sharper, and lower-noise** over time.

## Operating mode

- Prioritize correctness, coherence, and maintainability over new roadmap expansion.
- Prefer deleting, merging, or rewriting low-value content instead of preserving redundant guidance.
- Keep public claims aligned with what the repository actually implements and validates.
- Avoid over-engineering new automation, process, or tooling layers.

## Source of truth

1. **Active OpenSpec change** in `openspec/changes/<name>/`
2. **Current capability specs** in `openspec/specs/`
3. **Repository code and configuration**
4. **Project instruction files** (`AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`)

Local-only overrides such as `CLAUDE.local.md` are allowed for personal use, but they are **not** authoritative and should not be committed.

## Required workflow

### 1. Explore before guessing

Use `/opsx:explore` for open-ended investigation, architecture trade-offs, and requirement clarification.

### 2. Propose before non-trivial changes

Use `/opsx:propose <change-name>` for:

- repository-wide cleanup
- workflow changes
- public-surface repositioning
- feature or behavior changes
- multi-file refactors with architectural impact

### 3. Apply from explicit tasks

Use `/opsx:apply <change-name>` and implement directly from `tasks.md`. Work in dependency order and keep each task tightly scoped.

### 4. Review major workstreams

After major workflow, documentation, or architecture changes, run a review-oriented pass (`/review` or equivalent) before considering the workstream complete.

### 5. Archive finished changes

Use `/opsx:archive <change-name>` only after specs, code, docs, and validation results all agree.

## Repository priorities

### OpenSpec-first governance

- Do not reintroduce legacy `/specs` workflow guidance.
- Keep contributor docs aligned with `openspec/`.
- Preserve history through OpenSpec archive, not through duplicated meta-docs.

### Minimal, high-signal documentation

- Prefer one authoritative document per topic.
- Remove stale changelog pages, duplicated contribution docs, and generic AI boilerplate.
- Keep README and Pages useful for new users, not just comprehensive.

### Non-mutating automation

- CI should validate, not rewrite tracked files or commit back to the branch.
- Keep workflow triggers and jobs as small as possible while preserving confidence.

### Public-surface consistency

- README, GitHub Pages, release notes, and GitHub metadata must describe the same project.
- Remove unsupported roadmap promises and speculative positioning.

## Engineering invariants

- **Language/toolchain:** C++17, CUDA C++17, CMake 3.18+
- **GPU baseline:** CUDA 11.0+, Compute Capability 7.0+
- **Error handling:** `Result<T>` for fallible operations; no exceptions for control flow
- **CUDA safety:** use the project CUDA utilities/macros instead of silent failures
- **Resource management:** prefer RAII wrappers over raw ownership
- **Formatting:** `clang-format-18` with the repository `.clang-format`
- **Editor baseline:** `clangd` + `compile_commands.json`

### Validation commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
find . -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
  ! -path './build/*' | xargs clang-format-18 --dry-run --Werror
```

`nvcc` must be available for a real configure/build.

## Practical AI-tooling guidance

- **AGENTS.md**: repository-wide rules and architecture constraints
- **CLAUDE.md**: Claude/agent execution guidance for this repo
- **Copilot instructions**: concise code-generation and workflow constraints

Use a **long-running single session** for large repository hardening work. Avoid `/fleet` unless the work is clearly parallelizable and the higher token cost is justified.

Use subagents and research helpers selectively:

- good fit: independent audits, workflow inspection, broad doc inventories
- bad fit: tightly coupled refactors where one agent needs end-to-end context

Keep MCP usage narrow. Prefer built-in GitHub tooling and `gh` unless MCP gives a clear repository-specific win.

## Repository map

```text
openspec/                 OpenSpec config, active changes, archived changes, specs, schemas
include/tiny_llm/         Public headers
src/                      Host-side C++ implementation
kernels/                  CUDA kernels
tests/                    Unit and property tests
website/                  GitHub Pages site
.github/workflows/        CI, Pages, release automation
```

## Do not do these by default

- Do not add large new roadmap features just because they are easy to imagine.
- Do not preserve outdated docs “for completeness”.
- Do not commit personal-only tool config.
- Do not rely on CI to auto-fix formatting or repository drift.
- Do not split work across many stale branches without quick review and merge.
