---
layout: default
title: "Contributing — Tiny-LLM"
description: "How to contribute to Tiny-LLM project"
nav_order: 50
---

# Contributing to Tiny-LLM

Thank you for your interest in contributing to Tiny-LLM! We welcome contributions via Issues and Pull Requests.

---

## Development Workflow

This project follows **Spec-Driven Development (SDD)**. Before writing code, you must first update or create the relevant specification documents.

1. **Update/Create Specs**: Update or create product requirements, RFCs, or API definitions in `/specs`
2. Fork this repository
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Make your changes and ensure tests pass
5. Commit your changes: `git commit -m "feat: add your feature"`
6. Push to the branch: `git push origin feature/your-feature`
7. Create a Pull Request

> Detailed AI assistant workflow: see [AGENTS.md](https://github.com/LessUp/tiny-llm/blob/main/AGENTS.md)

---

## Build & Test

```bash
# Clone repository
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

---

## Requirements

- **NVIDIA GPU**: Compute Capability 7.0+ (Volta or newer)
- **CUDA Toolkit**: 11.0 or higher
- **CMake**: 3.18 or higher
- **C++ Compiler**: GCC 9+ or Clang 10+

---

## Code Standards

- **C++17** for all C++ code
- **CUDA 11.0+** for GPU kernels
- **CMake 3.18+** for build configuration
- Error handling via `Result<T>` monad (no exceptions for control flow)

---

## Getting Help

- **Issues**: [Report bugs or request features](https://github.com/LessUp/tiny-llm/issues)
- **Discussions**: [Ask questions](https://github.com/LessUp/tiny-llm/discussions)
- **Documentation**: [Read the docs](/docs/en/)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

[← Home](/) | [Documentation](/docs/en/)
