## ADDED Requirements

### Requirement: OpenSpec SHALL be the canonical change-management workflow
The repository SHALL use OpenSpec as the only normative workflow for non-trivial changes, repository restructuring, and cross-cutting cleanup. Contributor-facing guidance SHALL reference `openspec/` paths and OpenSpec commands instead of legacy `specs/` structures or ad hoc spec-first instructions.

#### Scenario: Cross-cutting change is prepared
- **WHEN** a contributor begins a repository-wide or behavior-affecting change
- **THEN** an active OpenSpec change SHALL exist before implementation proceeds

#### Scenario: Contributor guidance references the workflow
- **WHEN** a user reads project workflow documentation
- **THEN** the documentation SHALL reference only the canonical OpenSpec workflow and current repository structure

### Requirement: Repository structure SHALL stay coherent with documented ownership
The repository SHALL maintain a clear and current mapping between top-level directories, their purpose, and the documents that govern them. Legacy or misleading references to removed directories, obsolete ownership models, or personal-only configuration SHALL be removed from project-level guidance.

#### Scenario: Repository layout is described
- **WHEN** a top-level guide describes the repository structure
- **THEN** every referenced directory or file SHALL exist and match its documented purpose

#### Scenario: Personal configuration is distinguished
- **WHEN** the repository includes both project-level and personal/local configuration files
- **THEN** project-level guidance SHALL identify which files are authoritative for the repository and which are local-only

### Requirement: Repository automation SHALL be low-noise and non-mutating
Continuous integration, release, and documentation workflows SHALL validate repository state without auto-committing back to the branch. Workflow triggers, jobs, and output SHALL be minimized to the smallest set that preserves confidence in buildability, correctness, and public-site integrity.

#### Scenario: CI validates source quality
- **WHEN** a CI workflow checks formatting, buildability, or tests
- **THEN** it SHALL report failures directly instead of rewriting tracked files and creating commits

#### Scenario: Workflow inventory is reviewed
- **WHEN** a repository workflow provides no meaningful quality, release, or presentation value
- **THEN** it SHALL be removed, simplified, or merged into a higher-signal workflow

### Requirement: Repository documents SHALL be aggressively pruned for signal
Project-level documentation SHALL prioritize Tiny-LLM-specific, actionable guidance. Low-value, duplicated, stale, or generic documents SHALL be deleted, merged, or rewritten so the repository becomes smaller and more coherent over time.

#### Scenario: Duplicate guidance exists
- **WHEN** two or more documents explain the same workflow or project concept
- **THEN** the repository SHALL retain one authoritative version and remove or redirect the rest

#### Scenario: Stale content is discovered
- **WHEN** a document contains outdated structure, roadmap, or process claims
- **THEN** that content SHALL be corrected or removed before the cleanup change is considered complete
