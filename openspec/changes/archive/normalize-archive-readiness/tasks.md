## 1. Baseline audit

- [x] 1.1 Inventory the dirty working tree, legacy `specs/` references, active docs drift, workflow sprawl, and site/public-surface inconsistencies
- [x] 1.2 Run the current CMake configure/build/test flow and capture baseline failures, warnings, and environment/tooling gaps
- [x] 1.3 Inspect current GitHub metadata, Pages configuration, and workflow behavior so later cleanup is grounded in actual repository state

## 2. OpenSpec and governance normalization

- [x] 2.1 Align OpenSpec migration artifacts, repository structure references, and governance docs so OpenSpec is the only canonical workflow
- [x] 2.2 Rewrite contributor-facing workflow guidance to remove legacy `specs/` paths and define a closure-oriented development process
- [x] 2.3 Define the repository cleanup policy for low-value docs, non-mutating CI, and authoritative project-level guidance files

## 3. AI instruction and tooling baseline

- [x] 3.1 Curate `AGENTS.md` and create a project-level `CLAUDE.md` that capture Tiny-LLM-specific workflow, constraints, and review expectations
- [x] 3.2 Add concise project-level Copilot instructions and align them with OpenSpec, review, subagent, and autopilot usage guidance
- [x] 3.3 Standardize the developer tooling baseline for `clangd` + `compile_commands.json`, lightweight hooks, and optional MCP/plugin usage

## 4. Workflow and engineering configuration simplification

- [x] 4.1 Redesign CI so it validates formatting, buildability, and tests without mutating tracked files or auto-committing changes
- [x] 4.2 Simplify Pages and Release workflows, triggers, and supporting config to match a lower-noise archive-ready maintenance model
- [x] 4.3 Rationalize root engineering config such as `.gitignore`, local setup guidance, and related repository-quality files

## 5. Documentation and public-surface redesign

- [x] 5.1 Rewrite `README.md` and `README.zh-CN.md` around the final project positioning and remove stale roadmap or structure claims
- [x] 5.2 Prune or merge low-value docs, changelog surfaces, and supporting pages so the documentation tree becomes smaller and higher-signal
- [x] 5.3 Reposition GitHub Pages as a focused showcase/onboarding surface with stronger value proposition, architecture framing, and calls to action

## 6. Metadata alignment, bug remediation, and closure

- [x] 6.1 Update GitHub repository description, homepage URL, and topics with `gh` after README and Pages messaging are finalized
- [x] 6.2 Re-audit code/spec/doc drift and fix real defects or inconsistencies surfaced by the normalization work
- [x] 6.3 Re-run formatting, build, tests, and site validation, then resolve any regressions introduced by the cleanup
- [x] 6.4 Write the final lightweight closure playbook and prepare the change for archive once the repository is coherent
