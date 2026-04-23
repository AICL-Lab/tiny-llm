## Context

Tiny-LLM is a CUDA/C++ inference engine whose repository recently migrated from a custom `specs/` layout to OpenSpec. The migration created the right core structure, but several high-signal repository surfaces still point in different directions:

- contributor guidance still mixes legacy `specs/` instructions with OpenSpec
- GitHub Actions are more complex than the project's current maintenance intent and CI currently mutates the branch
- README, changelog, GitHub Pages, and GitHub metadata are not yet aligned around a focused, archive-ready project story
- AI-tooling guidance is fragmented across `AGENTS.md`, personal `CLAUDE.local.md`, OpenSpec Claude commands, and missing project-level Copilot/Claude guidance
- the working tree already contains user edits, so cleanup must be precise rather than destructive-by-default in execution order

This change is cross-cutting: it touches repository governance, documentation architecture, automation, public presentation, and the operator workflow used by AI agents during the project's final hardening phase.

## Goals / Non-Goals

**Goals:**
- Make OpenSpec the single authoritative planning and change-management spine.
- Reduce repository drift by removing stale guidance, duplicated docs, and low-value workflow behavior.
- Redesign public and contributor-facing surfaces so they reflect the actual project state and maintenance goals.
- Standardize the AI-assisted development model for this repository with a small number of project-specific instruction files.
- Establish a practical developer tooling baseline for a C++17/CUDA project, including LSP guidance and low-friction hooks.
- Leave the repository in a cleaner, lower-noise state suitable for final bug-fixing and later low-maintenance archival mode.

**Non-Goals:**
- Expanding Tiny-LLM into a broader feature roadmap or reviving speculative roadmap items.
- Introducing heavy new platform dependencies, large MCP/plugin stacks, or complex developer infrastructure.
- Rewriting stable core CUDA architecture without evidence from the later bug audit.
- Supporting every AI editor/runtime equally with bespoke configuration if shared guidance is sufficient.

## Decisions

### 1. Use a single umbrella OpenSpec change for repository normalization

This work spans multiple repository surfaces that need to stay coordinated. A single umbrella change keeps proposal, design, specs, and tasks unified while still allowing the implementation to proceed in narrow phases.

**Alternatives considered**
- **Many small changes**: lower local scope, but higher drift risk and weaker overall architecture control.
- **Ad hoc edits without a change**: faster initially, but contradicts the stated OpenSpec-first workflow.

### 2. Treat aggressive cleanup as the default content strategy

The user explicitly prefers aggressive cleanup. The implementation will therefore bias toward deleting, merging, or rewriting low-value documents and redundant workflow behavior instead of preserving them for historical completeness.

**Alternatives considered**
- **Preserve most files and only patch references**: safer short-term, but leaves the repository bloated and internally inconsistent.
- **Freeze current docs and only add new meta-docs**: increases duplication and does not solve drift.

### 3. Separate repository-governance concerns from inference-engine behavior

This change introduces new governance-oriented capabilities instead of overloading the existing `inference-engine` capability with repository process requirements. Code bug fixing discovered later should primarily align implementation to the existing engine spec unless a genuine behavior change is required.

**Alternatives considered**
- **Modify `inference-engine` now**: possible, but would mix repository-operating concerns into product behavior and create premature spec churn.

### 4. Make CI non-mutating and reduce automation noise

Validation pipelines should check quality and fail loudly; they should not auto-format files and commit back to the branch. This reduces surprise, avoids noisy workflow loops, and better fits a final-hardening repository.

**Alternatives considered**
- **Keep auto-format-on-push**: convenient for style drift, but high-noise and risky during stabilization.
- **Add more automation layers**: likely over-engineering for a project moving toward low maintenance.

### 5. Reframe GitHub Pages and README surfaces around credibility and conversion

The public project surface should do four things well: explain what Tiny-LLM is, show why it matters, establish its implementation maturity, and guide the right next step. Pages should not exist merely as a prettier README mirror.

**Alternatives considered**
- **Keep the current docs-heavy site mostly intact**: less work, but still under-optimized for attracting and orienting new users.
- **Remove Pages entirely**: would simplify maintenance, but loses a valuable project showcase and homepage target for GitHub metadata.

### 6. Use a minimal, project-specific AI tooling stack

The canonical project instruction surfaces will be:
- OpenSpec artifacts
- `AGENTS.md`
- project `CLAUDE.md`
- project-level Copilot instructions

Supplementary tool-specific configuration should only be added when it produces a concrete, repository-specific gain.

**Alternatives considered**
- **Many parallel instruction files for every tool**: higher maintenance, more drift, weak signal.
- **No project-level AI guidance**: leaves agents relying on generic behavior and increases repository-specific mistakes.

### 7. Standardize on clangd plus compile_commands.json for LSP

For this C++17/CUDA repository, `clangd` backed by `compile_commands.json` is the most portable baseline across editors and AI tooling ecosystems. This becomes the default recommendation unless a tool requires a thin compatibility layer.

**Alternatives considered**
- **Tool-specific LSP setups**: more fragmentation and duplicated guidance.
- **No explicit LSP baseline**: leads to inconsistent indexing and weaker editing assistance.

### 8. Keep MCP optional and narrow

MCP should be used only where it materially improves repository operations over existing built-in tools such as `gh`, GitHub integrations, or local CLI commands. This avoids context bloat and unnecessary runtime complexity.

**Alternatives considered**
- **Broad MCP adoption**: potentially powerful, but high context and maintenance overhead for limited payoff in this repository.

### 9. Update GitHub metadata only after content surfaces stabilize

Repository description, homepage URL, and topics should be updated with `gh` after README and Pages positioning are finalized so the public metadata points at the correct, polished destination.

**Alternatives considered**
- **Update metadata first**: creates temporary mismatch between GitHub summary and repo/site content.

## Risks / Trade-offs

- **Dirty working tree conflicts** -> Reconcile user-owned changes first and avoid blanket rewrites in touched files.
- **Cleanup scope becomes too broad** -> Execute in dependency-aware phases and define explicit completion criteria per workstream.
- **Accidentally removing useful historical context** -> Preserve only high-signal history through OpenSpec archive, meaningful changelog entries, and curated release notes.
- **Workflow simplification breaks expected automation** -> Re-validate build, tests, and Pages behavior after each major automation change.
- **Tooling guidance becomes too generic** -> Keep instructions tied to Tiny-LLM's actual stack and maintenance goal.
- **GitHub metadata drifts again later** -> Make metadata alignment part of the final closure playbook.

## Migration Plan

1. Reconcile the current dirty repository state and establish the baseline audit.
2. Normalize OpenSpec and contributor-governance surfaces first so later edits follow one workflow.
3. Simplify workflows, hooks, and engineering config before relying on them for further cleanup.
4. Redesign README, changelog policy, Pages content, and GitHub metadata as one public-surface pass.
5. Standardize AI/tooling guidance and then run the repository-wide bug/drift audit.
6. Finish with a closure playbook that defines the lightweight future maintenance path.

## Open Questions

- Which existing website/doc pages survive aggressive cleanup versus being merged into fewer, higher-signal entry points?
- Which GitHub topics best balance discoverability with accuracy for a CUDA/C++ inference engine nearing archival mode?
