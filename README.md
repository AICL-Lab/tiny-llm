# Tiny-LLM Inference Engine

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![Release](https://img.shields.io/github/v/release/LessUp/tiny-llm?include_prereleases)](https://github.com/LessUp/tiny-llm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

English | [简体中文](README.zh-CN.md) | [Documentation](https://lessup.github.io/tiny-llm/) | [API Reference](https://lessup.github.io/tiny-llm/docs/en/API)

A lightweight, high-performance CUDA C++ inference engine for Transformer models, featuring W8A16 quantization, efficient KV Cache management, and optimized CUDA kernels.

---

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| **W8A16 Quantization** | INT8 weights + FP16 activations for ~50% memory reduction | ✅ Stable |
| **KV Cache Management** | Efficient incremental decoding with sequence management | ✅ Stable |
| **Optimized CUDA Kernels** | Tensor Core INT8, shared memory tiling, warp shuffle | ✅ Stable |
| **Sampling Strategies** | Greedy, Temperature, Top-k, Top-p decoding | ✅ Stable |
| **Comprehensive Testing** | GoogleTest + RapidCheck property-based tests | ✅ Stable |
| **Bilingual Documentation** | Full documentation in English and Chinese | ✅ Complete |

### Roadmap

| Feature | Status | Target |
|---------|--------|--------|
| GGUF Runtime Loading | 🚧 Planned | v2.1 |
| PagedAttention | 📋 Planned | v2.2 |
| Speculative Decoding | 🔬 Research | v2.3 |
| FP8 Support | 🔬 Research | v3.0 |
| Multi-GPU | 📋 Planned | v3.0 |

---

## 🚀 Quick Start

### Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| NVIDIA GPU | SM 7.0 (Volta) | SM 8.0+ (Ampere+) |
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| C++ Compiler | GCC 9+ / Clang 10+ | GCC 11+ |

### Installation

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

### Usage Example

```cpp
#include <tiny_llm/inference_engine.h>

ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

auto engine = InferenceEngine::load("model.bin", config).value();

GenerationConfig gen;
gen.max_new_tokens = 256;
gen.temperature = 0.7f;

auto output = engine.generate({1, 15043, 29892}, gen);
```

---

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| [📖 Documentation](https://lessup.github.io/tiny-llm/) | Project documentation homepage |
| [🚀 Quick Start](https://lessup.github.io/tiny-llm/docs/en/QUICKSTART) | Installation and usage guide |
| [🏗️ Architecture](https://lessup.github.io/tiny-llm/docs/en/ARCHITECTURE) | System design and components |
| [📘 API Reference](https://lessup.github.io/tiny-llm/docs/en/API) | Complete API documentation |
| [⚡ Benchmarks](https://lessup.github.io/tiny-llm/docs/en/BENCHMARKS) | Performance benchmarks |
| [🔧 Troubleshooting](https://lessup.github.io/tiny-llm/docs/en/TROUBLESHOOTING) | Common issues and solutions |
| [📝 Changelog](changelog/) | Version history |

---

## 🔌 GPU Support

| Architecture | Compute Capability | Status |
|--------------|-------------------|--------|
| Volta | SM 7.0, 7.5 | ✅ Supported |
| Turing | SM 7.5 | ✅ Supported |
| Ampere | SM 8.0, 8.6 | ✅ Optimized |
| Ada Lovelace | SM 8.9 | ✅ Optimized |
| Hopper | SM 9.0 | ✅ Supported |

---

## 📁 Project Structure

```
tiny-llm/
├── include/tiny_llm/    # Public headers
├── kernels/             # CUDA kernels (.cu, .cuh)
├── src/                 # Host-side implementation (.cpp)
├── tests/               # Unit & property tests
├── docs/                # Documentation (EN/ZH)
│   ├── en/              # English docs
│   └── zh/              # Chinese docs
├── changelog/           # Changelog (EN/ZH)
└── CMakeLists.txt       # Build configuration
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contributor Setup

```bash
# Fork and clone
git clone https://github.com/your-username/tiny-llm.git
cd tiny-llm

# Build with tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)

# Run checks
ctest --output-on-failure
clang-format -i src/*.cpp tests/*.cu
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and [vLLM](https://github.com/vllm-project/vllm)
- Built with [GoogleTest](https://github.com/google/googletest) and [RapidCheck](https://github.com/emil-e/rapidcheck)

---

<p align="center">
  <a href="https://lessup.github.io/tiny-llm/">Documentation</a> •
  <a href="https://github.com/LessUp/tiny-llm/releases">Releases</a> •
  <a href="https://github.com/LessUp/tiny-llm/issues">Issues</a>
</p>

<p align="center">
  Made with ❤️ by the Tiny-LLM team
</p>
