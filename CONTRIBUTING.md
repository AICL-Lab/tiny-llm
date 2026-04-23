# Contributing to Tiny-LLM

Tiny-LLM accepts focused issues and pull requests, but the repository is now optimized for **coherent hardening work**, not broad roadmap expansion. Keep changes small, reviewable, and anchored in the current architecture.

## Core rules

1. **OpenSpec first.** Non-trivial changes start in `openspec/`, not in ad hoc docs or code-only branches.
2. **One authoritative source per topic.** Prefer deleting or merging stale docs over adding parallel guidance.
3. **Short-lived branches.** Avoid long-lived local/cloud drift; merge or rebase frequently.
4. **CI validates, contributors fix.** Do not rely on automation to rewrite tracked files for you.

## Required workflow

### 1. Explore when the scope is unclear

Use `/opsx:explore` for investigation, trade-off analysis, and requirement clarification.

### 2. Propose before cross-cutting implementation

Use `/opsx:propose <change-name>` for repository-wide cleanup, new behavior, or architectural changes. The proposal should produce:

- `proposal.md`
- `design.md`
- `specs/.../spec.md`
- `tasks.md`

### 3. Implement against explicit tasks

Use `/opsx:apply <change-name>` and work directly from `tasks.md`. Finish tasks in dependency order and keep edits tightly scoped.

### 4. Review before concluding major workstreams

Use `/review` or an equivalent review pass after major workflow, documentation, or architecture changes. Do not batch unrelated cleanup into one unreviewed sweep.

### 5. Archive completed changes

Use `/opsx:archive <change-name>` after specs, code, and docs are aligned and the repository is ready to preserve the change in `openspec/changes/archive/`.

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

## Repository hygiene

- Keep README, GitHub Pages, and GitHub metadata consistent with the actual shipped engine.
- Treat low-value changelog entries, duplicated workflow docs, and stale roadmap promises as cleanup targets.
- If a document references removed directories or obsolete workflow commands, fix or delete it before merging.

## Low-maintenance flow

When the repository only needs focused follow-up work, use this loop:

1. Keep one active OpenSpec change per meaningful workstream.
2. Prefer short-lived branches or a single coordinated branch over parallel stale branches.
3. Use review checkpoints after each major workstream instead of saving all review until the end.
4. Archive the change as soon as specs, code, and docs converge so the repository returns to a calm baseline quickly.

## Project-specific guidance

- **Repository workflow and architecture constraints:** [AGENTS.md](AGENTS.md)
- **Current capabilities and requirements:** `openspec/specs/`
- **Active and archived change history:** `openspec/changes/`

Use [Conventional Commits](https://www.conventionalcommits.org/) where practical (`feat:`, `fix:`, `docs:`, `ci:`, `refactor:`), but prioritize clear, reviewable changes over ceremony.
