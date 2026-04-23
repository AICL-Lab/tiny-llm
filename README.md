# Tiny-LLM Inference Engine

> CUDA-native C++ inference engine for focused Transformer deployments.

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![Release](https://img.shields.io/github/v/release/LessUp/tiny-llm?include_prereleases&label=version)](https://github.com/LessUp/tiny-llm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[English](README.md) • [简体中文](README.zh-CN.md) • [Documentation](https://lessup.github.io/tiny-llm/) • [Architecture](https://lessup.github.io/tiny-llm/docs/en/ARCHITECTURE) • [Changelog](https://lessup.github.io/tiny-llm/changelog/)

---

## Overview

Tiny-LLM is a lightweight inference engine for Transformer workloads on NVIDIA GPUs. It is built around a focused CUDA/C++ stack: W8A16 quantization, explicit KV cache management, hand-tuned kernels, and a repository workflow designed to keep the codebase coherent as it hardens.

## Why Tiny-LLM

- **W8A16 quantization** with INT8 weights and FP16 activations
- **Explicit KV cache management** for incremental decoding
- **CUDA-native kernels** with shared-memory and warp-level optimization patterns
- **Minimal runtime surface** with `spdlog` as the primary runtime dependency
- **OpenSpec-governed development** so architecture, docs, and changes stay aligned

## Core capabilities

| Capability | What it covers | Status |
|---|---|---|
| W8A16 inference path | Quantized weights, FP16 activations, CUDA kernels | Stable |
| KV cache manager | Sequence allocation, growth, and release | Stable |
| Sampling utilities | Greedy, temperature, top-k, top-p | Stable |
| Error handling | `Result<T>`-based fallible APIs | Stable |
| Test strategy | GoogleTest + RapidCheck coverage for core paths | Active |

## Build from source

Tiny-LLM requires a working CUDA toolchain (`nvcc` on `PATH` or an equivalent configured install).

| Component | Minimum |
|---|---|
| NVIDIA GPU | Compute Capability 7.0+ |
| CUDA Toolkit | 11.0+ |
| CMake | 3.18+ |
| C++ Compiler | GCC 9+ or Clang 10+ |

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
```

## Minimal usage example

```cpp
#include <tiny_llm/inference_engine.h>

int main() {
    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;

    auto engine = InferenceEngine::load("model.bin", config).value();

    GenerationConfig gen;
    gen.max_new_tokens = 256;
    gen.temperature = 0.7f;
    gen.top_p = 0.9f;

    auto output = engine.generate({1, 15043, 29892}, gen);
    (void)output;
}
```

## Repository map

```text
openspec/                 Current specs, active changes, archived changes, schemas
include/tiny_llm/         Public headers
src/                      Host-side C++ implementation
kernels/                  CUDA kernels
tests/                    Unit and property tests
website/                  GitHub Pages site and public docs
.github/workflows/        CI, Pages, and release automation
```

## Documentation and specs

- **Project site:** https://lessup.github.io/tiny-llm/
- **Architecture guide:** https://lessup.github.io/tiny-llm/docs/en/ARCHITECTURE
- **Developer guide:** https://lessup.github.io/tiny-llm/docs/en/DEVELOPER
- **OpenSpec source of truth:** `openspec/specs/`
- **Change history:** `openspec/changes/`

## Contributing

Tiny-LLM uses an OpenSpec-first workflow for non-trivial changes.

- Start with [CONTRIBUTING.md](CONTRIBUTING.md)
- Read repository-specific constraints in [AGENTS.md](AGENTS.md)
- Use `/opsx:propose`, `/opsx:apply`, and `/opsx:archive` for structured work

## License

Tiny-LLM is released under the [MIT License](LICENSE).
