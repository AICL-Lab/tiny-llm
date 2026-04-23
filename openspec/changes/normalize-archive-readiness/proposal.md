## Why

Tiny-LLM has completed a partial migration to OpenSpec, but the repository still behaves like a project with mixed governance models, duplicated documentation layers, and overbuilt automation. Before the project can be finished strongly and moved toward low-maintenance archival mode, its repository structure, contributor workflow, public presentation, and AI-assisted development guidance need to be normalized around a smaller, more coherent operating model.

## What Changes

- Create an OpenSpec-driven cleanup program that governs repository structure, document quality, and archive-readiness.
- Replace remaining legacy `specs/` references and mixed workflow guidance with a single OpenSpec-first development flow.
- Redesign AI collaboration guidance around project-specific `AGENTS.md`, project-level `CLAUDE.md`, and Copilot instructions, while keeping tool surface area minimal.
- Simplify GitHub Actions so CI validates quality instead of mutating the branch, and reduce noisy or redundant workflow behavior.
- Reposition GitHub Pages, README surfaces, changelog usage, and GitHub metadata so the project presents a focused, credible public surface.
- Define the preferred local development/tooling baseline for this C++/CUDA repository, including hooks, LSP strategy, review flow, and the trade-offs around MCP/plugins.
- Audit and fix repository drift, stale claims, and real defects uncovered during normalization.

## Capabilities

### New Capabilities
- `repository-governance`: Defines the required repository structure, OpenSpec-first governance, documentation quality bar, and workflow simplification rules for Tiny-LLM.
- `ai-assisted-development`: Defines the project-specific operating model for OpenSpec, AGENTS/CLAUDE/Copilot guidance, review checkpoints, subagent usage, and low-noise automation.
- `project-presentation`: Defines how README, GitHub Pages, changelog surfaces, and GitHub metadata should present the project to new users and future maintainers.

### Modified Capabilities
- None.

## Impact

- Affects `openspec/`, `AGENTS.md`, contributor docs, README surfaces, website content, and `.github/workflows/*`.
- Introduces project-level AI guidance artifacts and a clearer local development baseline.
- May remove low-value or duplicated documents, changelog entries, and workflow behavior as part of aggressive cleanup.
- Drives subsequent bug-fixing and repository hardening work from a single OpenSpec change instead of ad hoc edits.
