# Getting Started

Welcome to Tiny-LLM! This guide will help you get up and running quickly.

## What is Tiny-LLM?

Tiny-LLM is a minimal, educational LLM inference engine implemented in CUDA C++. It provides:

- **W8A16 Quantization**: INT8 weight quantization with 16-bit activations
- **CUDA Kernels**: Optimized kernels for attention, FFN, and more
- **KV Cache**: Efficient autoregressive generation with pre-allocated cache
- **No-Exception Design**: `Result\<T\>` based error propagation

## Prerequisites

Before you begin, ensure you have:

- **CUDA Toolkit** 11.0 or higher
- **CMake** 3.18 or higher
- **C++17** compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- **NVIDIA GPU** with Compute Capability 7.0+ (Volta or newer)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/shane0/tiny-llm.git
cd tiny-llm

# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure
```

## Next Steps

- [Installation Guide](/en/guide/installation) - Detailed installation instructions
- [Quick Start Tutorial](/en/guide/quickstart) - Build your first inference
- [Architecture Overview](/en/guide/architecture) - Understand the internals
