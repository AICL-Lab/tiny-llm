# Project Philosophy: Spec-Driven Development (SDD)

This project strictly follows the **Spec-Driven Development (SDD)** paradigm. All code implementations must use the specifications in the `/specs` directory as the single source of truth.

---

## Directory Context

| Directory | Purpose | File Types |
|-----------|---------|------------|
| `/specs/product/` | Product feature definitions and acceptance criteria | `.md` |
| `/specs/rfc/` | Technical design documents and architecture decisions | `NNNN-title.md` |
| `/specs/api/` | API interface definitions (human and machine readable) | `.yaml`, `.json` |
| `/specs/db/` | Model and data schema definitions | `.dbml`, `.sql` |
| `/specs/testing/` | BDD test case specifications | `.feature` |
| `/docs/` | Developer and user documentation | `.md` (EN/ZH) |
| `/website/` | GitHub Pages website source (Jekyll) | `.yml`, `.md`, `.html`, `.json` |

---

## AI Agent Workflow Instructions

When you (the AI agent) are asked to develop a new feature, modify an existing feature, or fix a bug, **you must strictly follow this workflow without skipping any steps**:

### Step 1: Review Specs

Before writing any code, first read the relevant documents in `/specs`:

1. Read the product spec in `/specs/product/` for feature requirements
2. Read the relevant RFC in `/specs/rfc/` for architecture decisions
3. Read the API definition in `/specs/api/` for interface contracts
4. Check `/specs/db/` for data model constraints

**If the user's request conflicts with existing specs**, stop immediately and point out the conflict. Ask the user whether to update the spec first.

### Step 2: Spec-First Update

If this is a new feature, or if existing interfaces/data structures need to change:

1. **Propose spec changes first** - Create or modify files in `/specs/`:
   - New feature: `specs/product/<feature-name>.md` + `specs/rfc/NNNN-<title>.md`
   - API changes: Update `specs/api/*.yaml`
   - Data model changes: Update `specs/db/*.dbml`
   - Test requirements: Create `specs/testing/<feature>.feature`

2. **Wait for user confirmation** before entering the code implementation phase

### Step 3: Implementation

When writing code, **100% adhere to the spec definitions**:

- Use exact variable naming from specs
- Follow API paths and data types from `specs/api/`
- Respect constraints defined in `specs/db/`
- Implement acceptance criteria from `specs/product/`

**Do not add features not defined in the specs (No Gold-Plating).**

### Step 4: Test Against Spec

Write tests based on the acceptance criteria in `/specs`:

1. Map each acceptance criterion to a test case
2. Cover all boundary conditions described in specs
3. Update `specs/testing/*.feature` if new test scenarios are needed

---

## Code Generation Rules

| Rule | Description |
|------|-------------|
| API Changes | Must synchronously update `/specs/api/*.yaml` |
| New Types | Must be defined in `/specs/db/schema-v1.dbml` |
| New Features | Require product spec in `/specs/product/` |
| Architecture Changes | Require RFC in `/specs/rfc/` |
| Test Requirements | Must be defined in `/specs/testing/` before implementation |

---

## Spec File Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Product | `<feature-name>.md` | `tiny-llm-inference-engine.md` |
| RFC | `<NNNN>-<short-title>.md` | `0001-core-architecture.md` |
| API | `<feature-name>.yaml` | `inference-engine.yaml` |
| DB Schema | `schema-v<version>.dbml` | `schema-v1.dbml` |
| Testing | `<feature-name>.feature` | `inference-engine.feature` |

---

## Current Project Context

### Tiny-LLM Inference Engine

A lightweight, high-performance CUDA C++ inference engine for Transformer models.

**Key Components:**

| Component | Location | Spec |
|-----------|----------|------|
| InferenceEngine | `include/tiny_llm/inference_engine.h` | `specs/api/inference-engine.yaml` |
| KVCacheManager | `include/tiny_llm/kv_cache.h` | `specs/rfc/0001-core-architecture.md` |
| W8A16 MatMul | `kernels/w8a16_matmul.cu` | `specs/rfc/0001-core-architecture.md` |
| Model Types | `include/tiny_llm/types.h` | `specs/db/schema-v1.dbml` |

**Architecture Constraints:**

- CUDA 11.0+, C++17, CMake 3.18+
- Compute Capability 7.0+ (Volta or newer)
- W8A16 quantization: INT8 weights, FP16 activations
- Error handling via `Result<T>` monad (no exceptions for control flow)

---

## Why This Matters

| Benefit | Explanation |
|---------|-------------|
| **Prevents AI hallucination** | Forcing the first step to read `/specs` anchors the AI's thinking scope |
| **Enforces modification path** | Declaring "modify specs before code" ensures document-code synchronization |
| **Improves PR quality** | Implementation aligns with business logic because it follows acceptance criteria |
| **Enables parallel work** | Specs allow multiple agents to work independently with clear contracts |

---

## Quick Reference

```bash
# Read all specs before starting work
cat specs/product/*.md      # Product requirements
cat specs/rfc/*.md          # Architecture decisions
cat specs/api/*.yaml        # API contracts
cat specs/db/*.dbml         # Data models
cat specs/testing/*.feature # Test specifications

# After code changes, verify specs are updated
git diff specs/             # Should show corresponding changes
```

---

## Related Documents

- [Contributing Guide](CONTRIBUTING.md) - Development workflow and code standards
- [Documentation](docs/) - User and developer documentation
- [Changelog](changelog/) - Version history
