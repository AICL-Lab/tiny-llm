---
layout: home

hero:
  name: "Tiny-LLM"
  text: "Educational CUDA LLM Inference"
  tagline: A minimal, educational LLM inference engine in CUDA C++ with INT8 quantization
  image:
    src: /images/logo.svg
    alt: Tiny-LLM Logo
  actions:
    - theme: brand
      text: Get Started
      link: /en/guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/LessUp/tiny-llm

features:
  - icon: "🚀"
    title: High Performance
    details: Optimized CUDA kernels for attention, FFN, and quantized inference with W8A16 quantization
  - icon: "📚"
    title: Educational
    details: Clean, well-documented codebase designed for learning LLM inference internals
  - icon: "📊"
    title: INT8 Quantization
    details: Per-group INT8 quantization for memory efficiency with minimal accuracy loss
  - icon: "🔐"
    title: No-Exception Design
    details: Result based error propagation for predictable and safe error handling
  - icon: "🧠"
    title: KV Cache
    details: Pre-allocated KV cache manager for efficient autoregressive generation
  - icon: "🌐"
    title: Cross-Platform
    details: Cross-platform support with CMake build system for Linux and Windows
---

## Key Features

### W8A16 Quantization
Per-group INT8 weight quantization with FP16 activation. Reduces memory by ~45% while maintaining accuracy.

### Result Error Handling
No exceptions thrown. All errors propagated through Result type for predictable control flow.

### KV Cache Management
Pre-allocated cache slots for efficient autoregressive generation with configurable cache size.

## Quick Start

```bash
# Clone
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm

# Build (requires CUDA 11.0+)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# Test
ctest --test-dir build --output-on-failure
```

## Learn More

| Resource | Description |
|----------|-------------|
| [Getting Started](/en/guide/getting-started) | Build and run your first inference |
| [Architecture](/en/architecture/overview) | Core components and data flow |
| [Quantization](/en/architecture/quantization) | W8A16 quantization details |
| [KV Cache](/en/architecture/kv-cache) | Cache management strategies |
| [API Reference](/en/api/) | Complete API documentation |
