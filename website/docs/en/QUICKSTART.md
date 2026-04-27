---
layout: default
title: "Quick Start — Tiny-LLM"
description: "Get started with Tiny-LLM inference engine in minutes"
nav_order: 2
---

# Quick Start

Get up and running with Tiny-LLM inference engine in minutes.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Example](#quick-example)
- [Model Format](#model-format)
- [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| C++ Compiler | GCC 9+ / Clang 10+ | GCC 11+ / Clang 14+ |
| GPU Compute Capability | SM 7.0 (Volta) | SM 8.0+ (Ampere+) |
| GPU Memory | 4 GB | 8 GB+ |

### Verify Your GPU

```bash
# Check CUDA version
nvcc --version

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
# Output should be 7.0 or higher
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm
```

### 2. Configure Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

#### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: Debug/Release/RelWithDebInfo |
| `BUILD_TESTS` | `ON` | Build test suite |
| `CUDA_ARCH` | `native` | Target CUDA architectures (e.g., `75;80;86`) |

### 3. Build

```bash
make -j$(nproc)
```

### 4. Run Tests

```bash
ctest --output-on-failure
```

### 5. Run Demo

```bash
./tiny_llm_demo
```

---

## Quick Example

### Complete Inference Example

```cpp
#include <tiny_llm/inference_engine.h>
#include <iostream>

int main() {
    // 1. Configure model
    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;
    config.num_heads = 32;
    config.num_kv_heads = 32;      // GQA: use 8 for modern models
    config.head_dim = 128;
    config.intermediate_dim = 11008;
    config.max_seq_len = 2048;
    config.rope_theta = 10000.0f;
    
    // 2. Load model
    auto result = InferenceEngine::load("path/to/model.bin", config);
    if (result.isErr()) {
        std::cerr << "Failed to load model: " << result.error() << std::endl;
        return 1;
    }
    auto engine = std::move(result.value());
    
    // 3. Configure generation
    GenerationConfig gen_config;
    gen_config.max_new_tokens = 256;
    gen_config.temperature = 0.7f;
    gen_config.top_p = 0.9f;
    gen_config.top_k = 50;
    gen_config.do_sample = true;
    
    // 4. Generate
    std::vector<int> prompt = {1, 15043, 29892};  // "Hello," tokens
    auto output = engine->generate(prompt, gen_config);
    
    // 5. Check statistics
    const auto& stats = engine->getStats();
    std::cout << "Generated " << stats.tokens_generated << " tokens\n"
              << "Speed: " << stats.tokens_per_second << " tok/s\n"
              << "Peak memory: " << stats.peak_memory_bytes / 1024 / 1024 << " MB\n";
    
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
cache_config.max_batch_size = 1;

// Use factory method for proper error handling
auto cache_result = KVCacheManager::create(cache_config);
if (cache_result.isErr()) {
    std::cerr << "Failed to create cache: " << cache_result.error() << std::endl;
    return 1;
}
auto kv_cache = std::move(cache_result.value());

// Allocate a sequence
auto seq_result = kv_cache->allocateSequence(1024);
if (seq_result.isErr()) {
    std::cerr << "Failed to allocate: " << seq_result.error() << std::endl;
    return 1;
}
int seq_id = seq_result.value();

// Use in transformer layers
for (auto& layer : layers) {
    layer.forward(hidden_states, *kv_cache, seq_id, position, stream);
}

// After all layers, advance sequence length
kv_cache->advanceSeqLen(seq_id, 1);

// Release when done
kv_cache->releaseSequence(seq_id);
```

---

## Model Format

### Custom Binary Format

Tiny-LLM currently uses a custom binary format with the following layout:

```
┌─────────────────┬─────────────────────────────────────┐
│ Header (256B)   │ magic, version, config              │
├─────────────────┼─────────────────────────────────────┤
│ Token Embedding │ [vocab_size, hidden_dim] FP16       │
├─────────────────┼─────────────────────────────────────┤
│ Layer 0 Weights │ Attention + FFN weights (INT8)      │
│                 │ Scales (FP16)                       │
├─────────────────┼─────────────────────────────────────┤
│ ...             │                                     │
├─────────────────┼─────────────────────────────────────┤
│ Layer N-1       │                                     │
├─────────────────┼─────────────────────────────────────┤
│ Output Norm     │ [hidden_dim] FP16                   │
│ LM Head         │ [hidden_dim, vocab_size] FP16       │
└─────────────────┴─────────────────────────────────────┘
```

### Creating Model Files

See [Developer Guide](DEVELOPER) for instructions on converting models to Tiny-LLM format.

---

## Next Steps

- **[Architecture](ARCHITECTURE)** — Understand system design and components
- **[API Reference](API)** — Complete API documentation
- **[Benchmarks](BENCHMARKS)** — Performance characteristics
- **[Troubleshooting](TROUBLESHOOTING)** — Common issues and solutions

---

**Languages**: [English](QUICKSTART) | [中文](../zh/QUICKSTART) | [API →](API)

[← Home](../../) | [Architecture →](ARCHITECTURE)
