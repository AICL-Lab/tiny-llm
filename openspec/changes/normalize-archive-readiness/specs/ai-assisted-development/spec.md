## ADDED Requirements

### Requirement: The repository SHALL provide a clear project-specific AI instruction hierarchy
Tiny-LLM SHALL define a compact, project-specific instruction hierarchy for AI-assisted development. `AGENTS.md`, project `CLAUDE.md`, OpenSpec artifacts, and project-level Copilot instructions SHALL be internally consistent and SHALL identify the repository's authoritative workflow, quality expectations, and scope boundaries.

#### Scenario: An AI agent starts work
- **WHEN** an AI agent reads repository guidance before making changes
- **THEN** it SHALL be able to identify the authoritative workflow, current architectural constraints, and where project-specific instructions live

#### Scenario: Local-only configuration coexists with project guidance
- **WHEN** a local or personal instruction file exists alongside project-level guidance
- **THEN** repository documentation SHALL make clear that local-only files do not override the project's authoritative rules

### Requirement: AI-assisted development SHALL follow an OpenSpec-centered operating model
AI-assisted work on Tiny-LLM SHALL use an OpenSpec-centered operating model: explore or propose first for non-trivial changes, apply against explicit tasks, and use review checkpoints before concluding major repository-wide edits. The operating model SHALL prefer long-running single-session execution over unnecessary parallel fleet usage and SHALL document when `/review` and subagents are appropriate.

#### Scenario: A non-trivial cleanup is requested
- **WHEN** an agent is asked to make a cross-cutting repository change
- **THEN** it SHALL anchor the work in an OpenSpec change with explicit tasks before broad implementation begins

#### Scenario: Review checkpoint is needed
- **WHEN** a major workstream such as workflow redesign, governance rewrite, or public-site repositioning is completed
- **THEN** the documented process SHALL require a review-oriented checkpoint before finalizing the workstream

### Requirement: Developer tooling guidance SHALL standardize LSP and right-size integrations
The project SHALL standardize on `clangd` plus `compile_commands.json` as the default LSP path for local development in this C++17/CUDA repository. Tooling guidance SHALL explain that this baseline is reusable across supported editors and AI coding tools, while MCP servers, plugins, and extra integrations SHALL remain optional and SHALL be adopted only when they provide clear repository-specific value.

#### Scenario: A developer configures local intelligence tools
- **WHEN** a developer or AI tool needs language-aware navigation for Tiny-LLM
- **THEN** the project guidance SHALL direct them to the canonical `clangd` plus `compile_commands.json` setup

#### Scenario: A new integration is considered
- **WHEN** a plugin, MCP server, or tool-specific integration is proposed
- **THEN** it SHALL be adopted only if its value exceeds the added maintenance and context cost
