---
layout: home

hero:
  name: "Tiny-LLM"
  text: "Technical Whitepaper"
  tagline: CUDA-native inference engine with W8A16 quantization
  image:
    src: /images/logo.svg
    alt: Tiny-LLM Logo
  actions:
    - theme: brand
      text: Architecture
      link: /en/architecture/
    - theme: alt
      text: Performance
      link: /en/performance/

features:
  - icon: "⚡"
    title: W8A16 Quantization
    details: INT8 weights with FP16 activations — 50% memory reduction with <1% accuracy loss
  - icon: "🔧"
    title: CUDA-Native Kernels
    details: Tensor Core INT8 matmul, optimized attention decode, warp-level primitives
  - icon: "📦"
    title: KV Cache Manager
    details: Pre-allocated slots, O(1) incremental decoding, sequence lifecycle management
  - icon: "🛡️"
    title: Result<T> Pattern
    details: No-exception error propagation for predictable control flow and safe resource handling
  - icon: "📊"
    title: OpenSpec Governance
    details: Requirements-driven development with automated test coverage validation
  - icon: "🧪"
    title: Property-Based Testing
    details: RapidCheck integration for invariant verification across input spaces
---

## Key Results

| Metric | Value | vs FP16 |
|--------|-------|---------|
| **Memory** | 7.8 GB | **50% ↓** |
| **Decode** | 85 tok/s | **9% ↑** |
| **Accuracy** | 9.12 ppl | 0.4% Δ |

*Benchmarks: LLaMA-7B, RTX 4090, INT8 weights*

## Architecture

```mermaid
flowchart LR
    A[Model File] --> B[GGUF Parser]
    B --> C[InferenceEngine]
    C --> D[Transformer Layers]
    D --> E[KV Cache]
    D --> F[W8A16 Matmul]
    F --> G[Sampling]
    G --> H[Output Tokens]

    style C fill:#00D4AA,stroke:#00C49A,color:#fff
    style F fill:#76B900,stroke:#5a8f00,color:#fff
```

## Quick Start

```bash
# Clone
git clone https://github.com/AICL-Lab/tiny-llm.git
cd tiny-llm

# Build (requires CUDA 11.0+)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# Test
ctest --test-dir build --output-on-failure
```

## Documentation

| Resource | Description |
|----------|-------------|
| [Architecture Overview](/en/architecture/) | System design and data flow |
| [W8A16 Quantization](/en/architecture/quantization) | Quantization scheme details |
| [CUDA Kernels](/en/architecture/cuda-kernels) | Kernel optimization techniques |
| [Performance](/en/performance/) | Benchmarks and profiling guides |
| [API Reference](/en/api/) | Complete API documentation |

## Core Components

| Component | Responsibility |
|-----------|----------------|
| `Result<T>` | No-exception error propagation |
| `ModelConfig` | Model hyperparameters (vocab_size, hidden_dim, etc.) |
| `QuantizedWeight` | INT8 weights with per-group scales |
| `TransformerLayer` | W8A16 quantized attention + FFN |
| `KVCacheManager` | Pre-allocated cache slots for sequences |
| `InferenceEngine` | Public API: load(), generate() |
