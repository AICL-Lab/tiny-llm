---
layout: default
title: "Quick Start — Tiny-LLM"
description: "Get started with Tiny-LLM inference engine"
nav_order: 1
---

# Quick Start

Get up and running with Tiny-LLM in minutes.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| CUDA Toolkit | 11.0+ | Required for compilation |
| CMake | 3.18+ | Build system |
| C++ Compiler | C++17 | GCC 9+, Clang 10+, MSVC 2019+ |
| NVIDIA GPU | SM 7.0+ | Volta architecture or newer |

### Verify GPU Support

Check your GPU compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# Should output 7.0 or higher
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm
```

### 2. Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: Debug/Release/RelWithDebInfo |
| `BUILD_TESTS` | `ON` | Build test suite |
| `CUDA_ARCH` | `native` | CUDA architecture (e.g., `75;80;86`) |

### 3. Run Tests

```bash
ctest --output-on-failure
```

### 4. Run Demo

```bash
./tiny_llm_demo
```

---

## Basic Usage

### Simple Inference

```cpp
#include <tiny_llm/inference_engine.h>
#include <iostream>

int main() {
    // Configure model
    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;
    config.num_heads = 32;
    config.max_seq_len = 2048;
    
    // Load model
    auto result = InferenceEngine::load("path/to/model.bin", config);
    if (result.isErr()) {
        std::cerr << "Failed to load model: " << result.error() << std::endl;
        return 1;
    }
    
    auto engine = std::move(result.value());
    
    // Configure generation
    GenerationConfig gen_config;
    gen_config.max_new_tokens = 100;
    gen_config.temperature = 0.7f;
    gen_config.top_p = 0.9f;
    gen_config.do_sample = true;
    
    // Generate
    std::vector<int> prompt = {1, 2, 3};  // Token IDs
    auto output = engine->generate(prompt, gen_config);
    
    // Print statistics
    const auto& stats = engine->getStats();
    std::cout << "Generated " << stats.tokens_generated 
              << " tokens in " << stats.decode_time_ms << " ms"
              << " (" << stats.tokens_per_second << " tok/s)" << std::endl;
    
    return 0;
}
```

### Using KV Cache Directly

```cpp
#include <tiny_llm/kv_cache.h>

// Create cache manager
KVCacheConfig cache_config;
cache_config.num_layers = 32;
cache_config.num_heads = 32;
cache_config.head_dim = 128;
cache_config.max_seq_len = 2048;

KVCacheManager kv_cache(cache_config);

// Allocate a sequence
auto seq_result = kv_cache.allocateSequence(1024);
if (seq_result.isErr()) {
    // Handle error
}
int seq_id = seq_result.value();

// Use in transformer layer
layer.forward(hidden_states, kv_cache, seq_id, position);

// Advance sequence length after all layers
kv_cache.advanceSeqLen(seq_id, 1);

// Release when done
kv_cache.releaseSequence(seq_id);
```

---

## Building Your Own Model Format

Currently, Tiny-LLM supports a custom binary format. Here's the expected layout:

```
Model File Format:
┌─────────────────┐
│  Header (256B)  │
│  - magic        │
│  - version      │
│  - config       │
├─────────────────┤
│  Embeddings     │
│  [V, H] FP16    │
├─────────────────┤
│  Layer 0        │
│  - Weights...   │
├─────────────────┤
│  ...            │
├─────────────────┤
│  Layer N        │
├─────────────────┤
│  Output Norm    │
│  LM Head        │
└─────────────────┘
```

Note: Full GGUF loading support is planned for a future release.

---

## Troubleshooting

### Build Issues

| Problem | Solution |
|---------|----------|
| `CUDA not found` | Set `CUDA_TOOLKIT_ROOT_DIR` or ensure `nvcc` is in PATH |
| `CMake version too old` | Upgrade CMake or use pip: `pip install cmake` |
| `C++17 not supported` | Upgrade your compiler |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `max_seq_len` or `num_layers` in config |
| `Illegal memory access` | Check model file format and dimensions |
| `Slow performance` | Ensure Release build: `-DCMAKE_BUILD_TYPE=Release` |

### Debug Build

For debugging:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)
CUDA_LAUNCH_BLOCKING=1 ./tiny_llm_demo
```

---

## Next Steps

- Read the [API Reference](API) for detailed API documentation
- Check the [Architecture](ARCHITECTURE) guide for system design
- Review [Contributing](../../CONTRIBUTING) to contribute to the project
- See [Changelog](../../changelog/) for version history

---

**Languages**: [English](QUICKSTART) | [中文](../zh/QUICKSTART)

[← Home](../../) | [Architecture](ARCHITECTURE) | [API Reference](API)
