# Tiny-LLM Inference Engine

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![Release](https://img.shields.io/github/v/release/LessUp/tiny-llm?include_prereleases)](https://github.com/LessUp/tiny-llm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

English | [简体中文](README.zh-CN.md) | [Documentation](https://lessup.github.io/tiny-llm/) | [API Reference](https://lessup.github.io/tiny-llm/docs/)

A lightweight CUDA C++ inference engine for experimenting with W8A16 quantization, KV Cache incremental decoding, and modular Transformer inference.

---

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **W8A16 Quantization** | INT8 weights + FP16 activations, ~50% memory reduction | ✅ Stable |
| **KV Cache** | Efficient incremental decoding with sequence management | ✅ Stable |
| **High-Performance Kernels** | Shared memory tiling, warp shuffle optimization | ✅ Stable |
| **Sampling Strategies** | Greedy, Temperature, Top-k, Top-p | ✅ Stable |
| **Comprehensive Testing** | GoogleTest + RapidCheck property-based tests | ✅ Stable |
| GGUF Runtime Loading | Load models at runtime | 🚧 Planned |
| PagedAttention | Efficient batching | 🚧 Evaluating |

---

## Quick Start

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
./tiny_llm_demo
```

**Requirements**: CUDA Toolkit 11.0+, CMake 3.18+, C++17 compiler, NVIDIA GPU (Compute Capability 7.0+)

---

## GPU Architecture Support

| Architecture | Compute Capability | Status |
|--------------|-------------------|--------|
| Volta | SM 7.0, 7.5 | ✅ Supported |
| Turing | SM 7.5 | ✅ Supported |
| Ampere | SM 8.0, 8.6 | ✅ Optimized |
| Ada Lovelace | SM 8.9 | ✅ Optimized |
| Hopper | SM 9.0 | ✅ Supported |

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  InferenceEngine                            │
├────────────────────────────────────────────────────────────┤
│  Model Loader ──► Weights (INT8 + FP16 scales)             │
├────────────────────────────────────────────────────────────┤
│  Transformer Layers × N                                    │
│  ├── Attention: W8A16 MatMul + KV Cache + RoPE + Mask      │
│  └── FFN: W8A16 MatMul + SwiGLU                            │
├────────────────────────────────────────────────────────────┤
│  Sampling: Greedy / Temperature / Top-k / Top-p            │
└────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
tiny-llm/
├── include/tiny_llm/    # Public headers
├── kernels/             # CUDA kernels (.cu, .cuh)
├── src/                 # Host-side implementation (.cpp)
├── tests/               # Unit & property tests
├── docs/                # Documentation (EN/ZH)
│   ├── en/              # English docs
│   └── zh/              # Chinese docs
└── changelog/           # Changelog (EN/ZH)
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [📚 Documentation](https://lessup.github.io/tiny-llm/) | Project documentation homepage |
| [🚀 Quick Start](https://lessup.github.io/tiny-llm/docs/en/QUICKSTART) | Installation and basic usage |
| [🏗️ Architecture](https://lessup.github.io/tiny-llm/docs/en/ARCHITECTURE) | System design and components |
| [📖 API Reference](https://lessup.github.io/tiny-llm/docs/en/API) | Complete API documentation |
| [📝 Changelog](https://lessup.github.io/tiny-llm/changelog/) | Version history |
| [🤝 Contributing](CONTRIBUTING.md) | Development guidelines |

---

## Performance Highlights

| Optimization | Technique | Benefit |
|--------------|-----------|---------|
| Memory | W8A16 Quantization | ~50% weight memory reduction |
| Compute | Tensor Core INT8 | Accelerated matrix multiplication |
| Memory Bandwidth | Kernel Fusion | Reduced data movement |
| Latency | KV Cache | O(1) incremental decoding |

---

## Current Status

**v2.0.1** — Core runtime, cache management, and test infrastructure are complete. The demo validates CUDA availability.

### Roadmap

- [ ] Full GGUF runtime loading support
- [ ] Configurable `group_size` for W8A16 quantization
- [ ] Paged Attention for efficient batching
- [ ] Speculative decoding

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and [vLLM](https://github.com/vllm-project/vllm)
- Built with [GoogleTest](https://github.com/google/googletest) and [RapidCheck](https://github.com/emil-e/rapidcheck)

---

<p align="center">
  Made with ❤️ by the Tiny-LLM team
</p>
