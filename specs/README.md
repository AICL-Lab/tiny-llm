# Specifications

This directory contains all specification documents for the Tiny-LLM project. Specs are the **single source of truth** for all implementation decisions.

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `product/` | Product feature definitions, requirements, and acceptance criteria |
| `rfc/` | Technical design documents and architecture decisions (RFC format) |
| `api/` | API interface definitions (YAML, human and machine readable) |
| `db/` | Database and model schema definitions |
| `testing/` | BDD-style test case specifications (.feature files) |

## Naming Conventions

- **Product specs**: `<feature-name>.md` (e.g., `tiny-llm-inference-engine.md`)
- **RFCs**: `<NNNN>-<short-title>.md` (e.g., `0001-core-architecture.md`)
- **API specs**: `<feature-name>.yaml` or `<feature-name>.json`
- **Test specs**: `<feature-name>.feature`

## Workflow

1. **New feature**: Create a product spec in `product/` and an RFC in `rfc/`
2. **API changes**: Update or create files in `api/`
3. **Test cases**: Define in `testing/` before implementation

See [AGENTS.md](../AGENTS.md) for the complete Spec-Driven Development workflow.
