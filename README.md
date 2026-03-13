# Tiny-LLM Inference Engine

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

English | [简体中文](README.zh-CN.md) | [Docs](https://lessup.github.io/tiny-llm/)

Tiny-LLM is a lightweight CUDA C++ inference engine for experimenting with W8A16 quantization, KV Cache incremental decoding, and modular Transformer inference.

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
```

## Read Next

- [Documentation Home](https://lessup.github.io/tiny-llm/)
- [API Reference](https://lessup.github.io/tiny-llm/docs/API)
- [Changelog](https://lessup.github.io/tiny-llm/changelog/)
- [Contributing Guide](CONTRIBUTING.md)

## License

MIT License.
