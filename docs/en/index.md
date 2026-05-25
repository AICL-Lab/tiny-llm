---
layout: home

hero:
  name: "Tiny-LLM"
  text: "CUDA-Native Inference Engine"
  tagline: Focused Transformer inference with W8A16 kernels, explicit KV cache management, and a smaller-maintenance repository.
  image:
    src: /images/logo.svg
    alt: Tiny-LLM Logo
  actions:
    - theme: brand
      text: Get Started
      link: /en/guide/getting-started
    - theme: alt
      text: API Reference
      link: /en/api/

features:
  - icon: "⚡"
    title: W8A16 Inference
    details: INT8 weights with FP16 activations for a compact CUDA runtime path.
  - icon: "📦"
    title: Binary Runtime Path
    details: "`InferenceEngine::load()` targets the supported binary runtime format used by the current engine."
  - icon: "🔎"
    title: GGUF Parsing Utilities
    details: GGUF parsing, metadata extraction, and tensor inspection remain available without pretending to be the runtime path.
  - icon: "🧠"
    title: Explicit KV Cache
    details: Pre-allocated sequence slots keep autoregressive decoding predictable.
  - icon: "🛡️"
    title: "Result<T> APIs"
    details: Host-side failures stay explicit and inspectable.
  - icon: "🧪"
    title: Tested Core
    details: GoogleTest and RapidCheck cover the core loader, cache, and generation surfaces.
---

## Runtime Surfaces

| Surface | Status | Notes |
|---------|--------|-------|
| Binary runtime loading | Supported | Used by `InferenceEngine::load()` |
| GGUF parsing and inspection | Supported | Use `GGUFParser` for metadata and tensor access |
| Direct GGUF runtime loading | Not supported | `.gguf` paths are rejected by the runtime loader |

## Architecture Sketch

```mermaid
flowchart LR
    A[Binary runtime model] --> B[InferenceEngine::load()]
    C[GGUF file] --> D[GGUFParser]
    B --> E[Transformer Layers]
    E --> F[KV Cache]
    E --> G[W8A16 MatMul]
    G --> H[Sampling]
    H --> I[Output Tokens]
    D --> J[Metadata / Tensor Inspection]

    style B fill:#00D4AA,stroke:#00C49A,color:#fff
    style G fill:#76B900,stroke:#5a8f00,color:#fff
    style D fill:#8B5CF6,stroke:#7c3aed,color:#fff
```

## Start Here

| Resource | Why it matters |
|----------|----------------|
| [Getting Started](/en/guide/getting-started) | Minimal build and first-run path |
| [Architecture](/en/architecture/) | Runtime structure and responsibilities |
| [API Reference](/en/api/) | Public headers and types |
| [Performance](/en/performance/) | Benchmarks and optimization notes |
| [GitHub Repository](https://github.com/AICL-Lab/tiny-llm) | Source, issues, and releases |
