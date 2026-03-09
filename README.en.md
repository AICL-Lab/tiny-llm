# Tiny-LLM Inference Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

[简体中文](README.md) | English

A lightweight LLM inference engine using CUDA C++ with W8A16 quantized inference.

## Features

- **W8A16 Quantization** — INT8 weights + FP16 activations, 50% VRAM reduction
- **Efficient CUDA Kernels** — Shared memory tiling, warp shuffle optimization
- **KV Cache Management** — Incremental decoding, avoids redundant computation
- **Multiple Sampling Strategies** — Greedy, temperature, top-k, top-p
- **Modular Design** — Easy to extend and customize

## Requirements

- CUDA Toolkit 11.0+, CMake 3.18+, C++17, GPU CC 7.0+

## Build & Run

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run tests
./tiny_llm_tests
```

## Usage

```cpp
#include "tiny_llm/inference_engine.h"
using namespace tiny_llm;

ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

auto engine = InferenceEngine::load("model.bin", config).value();

std::vector<int> prompt = {1, 15043, 29892};  // "Hello,"
GenerationConfig gen;
gen.max_new_tokens = 100;
gen.temperature = 0.7f;
gen.do_sample = true;

auto output = engine->generate(prompt, gen);
```

## Core Components

- **W8A16 Matmul** — INT8 weight × FP16 activation with in-kernel dequantization
- **KV Cache** — Pre-allocated GPU memory pool, multi-sequence support, auto-reclaim
- **Sampling** — Greedy, temperature, top-k, top-p strategies

## Project Structure

```
├── include/tiny_llm/     # Headers (types, result, cuda_utils, kv_cache, transformer, engine)
├── kernels/              # CUDA kernels (w8a16_matmul, attention, rmsnorm)
├── src/                  # Source files
├── tests/                # Google Test + RapidCheck
└── CMakeLists.txt
```

## License

MIT License
