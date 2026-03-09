# Tiny-LLM Inference Engine

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

English | [简体中文](README.zh-CN.md) | [Project Page](https://lessup.github.io/tiny-llm/)

A lightweight LLM inference engine using CUDA C++ with W8A16 quantized inference. Reduces model VRAM by 50% with INT8 weights + FP16 activations, supports KV Cache incremental decoding and multiple sampling strategies.

## Features

- **W8A16 Quantization** — INT8 weights + FP16 activations, 50% VRAM reduction, in-kernel dequantization
- **Efficient CUDA Kernels** — Shared memory tiling, warp shuffle reduction for matmul / attention / RMSNorm
- **KV Cache Management** — Pre-allocated GPU memory pool, incremental decoding, multi-sequence support
- **Multiple Sampling Strategies** — Greedy, temperature, top-k, top-p (nucleus)
- **Modular Design** — Clean separation of kernels, transformer layers, model loading, and generation
- **Engineering Quality** — CI pipeline, clang-format, RAII memory management, Result error handling

## Requirements

- CUDA Toolkit 11.0+, CMake 3.18+, C++17 compiler, GPU CC 7.0+ (Volta → Hopper)

## Build & Run

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Usage

```cpp
#include "tiny_llm/inference_engine.h"
using namespace tiny_llm;

ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

auto result = InferenceEngine::load("model.bin", config);
if (result.isErr()) {
    std::cerr << "Error: " << result.error() << std::endl;
    return 1;
}
auto engine = std::move(result.value());

GenerationConfig gen;
gen.max_new_tokens = 100;
gen.temperature = 0.7f;
gen.do_sample = true;

auto output = engine->generate({1, 15043, 29892}, gen);  // "Hello,"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    InferenceEngine                           │
│  ┌───────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │  Model    │  │  Transformer  │  │  Generation          │ │
│  │  Loader   │──▶  Layers       │──▶  (Sampling + Decode) │ │
│  └───────────┘  └───────┬───────┘  └──────────────────────┘ │
│                         │                                    │
│  ┌───────────┐  ┌───────▼───────┐  ┌──────────────────────┐ │
│  │  Stream   │  │  KV Cache     │  │  Result<T>           │ │
│  │  Pool     │  │  Manager      │  │  Error Handling      │ │
│  └───────────┘  └───────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     CUDA Kernels                             │
│  ┌──────────────┐  ┌───────────┐  ┌────────────────────┐    │
│  │ W8A16 MatMul │  │ Attention │  │ RMSNorm            │    │
│  │ (tiling +    │  │ (KV Cache │  │ (warp shuffle      │    │
│  │  dequant)    │  │  + mask)  │  │  reduction)        │    │
│  └──────────────┘  └───────────┘  └────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

- **W8A16 Matmul** — INT8 weight × FP16 activation with fused in-kernel dequantization (shared memory tiling)
- **Attention** — Prefill (multi-token, causal mask) + Decode (single-token, KV cache) modes
- **RMSNorm** — Warp shuffle reduction, numerically stable
- **KV Cache** — Pre-allocated GPU memory pool, multi-sequence support, stateless per-layer append + explicit advanceSeqLen
- **Sampling** — Greedy, temperature, top-k, top-p strategies with configurable repetition penalty

## Project Structure

```
tiny-llm/
├── include/tiny_llm/          # Public headers
│   ├── types.h                # ModelConfig, GenerationConfig, QuantizedWeight
│   ├── result.h               # Result<T> error handling (Rust-style)
│   ├── cuda_utils.h           # CUDA_CHECK, DeviceBuffer<T> RAII
│   ├── cuda_streams.h         # StreamPool
│   ├── kv_cache.h             # KVCacheManager
│   ├── model_loader.h         # Model loading
│   ├── transformer.h          # TransformerLayer
│   └── inference_engine.h     # InferenceEngine
├── kernels/                   # CUDA kernels
│   ├── w8a16_matmul.cu/.cuh   # W8A16 quantized matmul (tiling + fused dequant)
│   ├── attention.cu/.cuh      # Attention (prefill + decode, KV cache)
│   ├── rmsnorm.cu/.cuh        # RMSNorm (warp shuffle reduction)
│   ├── elementwise.cu/.cuh    # Elementwise ops (SiLU, residual add)
│   └── warp_utils.cuh         # Warp-level primitives
├── src/                       # Host source files
│   ├── inference_engine.cpp   # Engine main logic
│   ├── transformer.cpp        # Transformer forward pass
│   ├── kv_cache.cpp           # KV cache alloc / append / reclaim
│   ├── model_loader.cpp       # Model file loading
│   └── main.cpp               # Demo entry point
├── tests/                     # Google Test
└── CMakeLists.txt             # CMake build (v2.0.0, FetchContent GTest)
```

## GPU Support

| Architecture | Compute Capability | Example GPUs |
|-------------|-------------------|-------------|
| Volta | SM 7.0 | V100 |
| Turing | SM 7.5 | RTX 2080, T4 |
| Ampere | SM 8.0 / 8.6 | A100, RTX 3090 |
| Ada Lovelace | SM 8.9 | RTX 4090, L40 |
| Hopper | SM 9.0 | H100 |

## Testing

```bash
./tiny_llm_tests --gtest_filter="W8A16*"       # Quantized matmul
./tiny_llm_tests --gtest_filter="Attention*"    # Attention mechanism
./tiny_llm_tests --gtest_filter="KVCache*"      # Cache management
./tiny_llm_tests --gtest_filter="Integration*"  # End-to-end
```

| Test Suite | Coverage |
|-----------|---------|
| W8A16 MatMul | Quantization accuracy, tiling correctness, boundary sizes |
| Attention | Masked self-attention, KV cache append, prefill/decode |
| RMSNorm | Normalization invariants, numerical stability |
| KV Cache | Allocation, append, multi-sequence, advanceSeqLen |
| Transformer | Layer forward pass, weight loading |
| Integration | End-to-end prompt → generation |

## License

MIT License
