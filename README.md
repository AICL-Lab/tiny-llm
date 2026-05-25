# Tiny-LLM

> CUDA-native C++ inference engine for focused Transformer workloads.

[![CI](https://github.com/AICL-Lab/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/AICL-Lab/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/AICL-Lab/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://aicl-lab.github.io/tiny-llm/)
[![Release](https://img.shields.io/github/v/release/AICL-Lab/tiny-llm?include_prereleases&label=version)](https://github.com/AICL-Lab/tiny-llm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[English](README.md) • [简体中文](README.zh-CN.md) • [Documentation](https://aicl-lab.github.io/tiny-llm/) • [Architecture](https://aicl-lab.github.io/tiny-llm/en/architecture/) • [API](https://aicl-lab.github.io/tiny-llm/en/api/) • [Changelog](CHANGELOG.md)

---

## Overview

Tiny-LLM keeps the repository surface deliberately small: CUDA/C++17 kernels, W8A16 quantization, explicit KV cache management, and a narrow runtime path that is easier to audit and maintain.

## What is implemented

- **W8A16 inference path** with INT8 weights and FP16 activations
- **Explicit KV cache management** for autoregressive decoding
- **CUDA-native kernels** with shared-memory and warp-level optimization patterns
- **`Result<T>`-based fallible APIs** for host-side error propagation
- **GoogleTest + RapidCheck coverage** for core runtime paths

## Model-loading surfaces

- `InferenceEngine::load()` currently accepts the repository's supported **binary runtime format**.
- `GGUFParser` is available for **GGUF parsing, metadata extraction, and tensor inspection**.
- Direct GGUF runtime loading is **not** part of the current inference path.

## Build from source

Tiny-LLM requires a working CUDA toolchain (`nvcc` on `PATH` or an equivalent configured installation).

| Component | Minimum |
|---|---|
| NVIDIA GPU | Compute Capability 7.0+ |
| CUDA Toolkit | 11.0+ |
| CMake | 3.18+ |
| C++ Compiler | GCC 9+ or Clang 10+ |

```bash
git clone https://github.com/AICL-Lab/tiny-llm.git
cd tiny-llm

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
```

## Minimal usage example

```cpp
#include <iostream>
#include <tiny_llm/inference_engine.h>

int main() {
    using namespace tiny_llm;

    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;

    auto engine_result = InferenceEngine::load("model.bin", config);
    if (engine_result.isErr()) {
        std::cerr << engine_result.error() << '\n';
        return 1;
    }

    GenerationConfig gen;
    gen.max_new_tokens = 64;
    gen.temperature = 0.7f;
    gen.top_p = 0.9f;
    gen.do_sample = true;

    auto engine = std::move(engine_result.value());
    auto output = engine->generate({1, 15043, 29892}, gen);
    if (output.isErr()) {
        std::cerr << output.error() << '\n';
        return 1;
    }

    return 0;
}
```

## Repository map

```text
include/tiny_llm/         Public headers
src/                      Host-side C++ implementation
kernels/                  CUDA kernels
tests/                    Unit and property tests
docs/                     VitePress documentation site
.github/workflows/        CI, Pages, and release automation
CHANGELOG.md              Canonical tracked release history
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending changes. Keep changes focused, keep docs aligned with the real runtime surface, and keep the repository free of duplicate workflow scaffolding.

## License

Tiny-LLM is released under the [MIT License](LICENSE).
