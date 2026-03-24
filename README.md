# Tiny-LLM Inference Engine

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

English | [简体中文](README.zh-CN.md) | [Docs](https://lessup.github.io/tiny-llm/)

Tiny-LLM is a lightweight CUDA C++ inference engine for experimenting with W8A16 quantization, KV Cache incremental decoding, and modular Transformer inference.

Current status: the core runtime, cache flow, and test scaffolding are implemented, but the repository is still experimental. The default demo binary currently reports CUDA readiness rather than providing a polished end-to-end CLI, and runtime GGUF loading is not supported yet.

## Repository Overview

- W8A16 quantized inference with INT8 weights and FP16 activations
- CUDA kernels for matmul, attention, RMSNorm, and elementwise ops
- Host-side modules for model loading, transformer execution, generation, and cache management
- Dedicated docs site for quick start, API reference, changelog, and contribution notes

## Quick Start

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
./tiny_llm_demo
```

Notes:
- A working CUDA toolkit with `nvcc` is required to configure/build this project.
- The demo currently validates CUDA availability and prints runtime capability information.
- `InferenceEngine::load()` currently supports the project test binary path via `ModelLoader::loadBin()`; `.gguf` runtime loading is not wired up yet.

## Read Next

- [Documentation Home](https://lessup.github.io/tiny-llm/)
- [API Reference](https://lessup.github.io/tiny-llm/docs/API)
- [Changelog](https://lessup.github.io/tiny-llm/changelog/)
- [Contributing Guide](CONTRIBUTING.md)

## License

MIT License.
