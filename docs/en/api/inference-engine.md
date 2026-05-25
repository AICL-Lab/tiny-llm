---
layout: doc
title: "Inference Engine API"
description: "Public runtime API for Tiny-LLM"
---

# Inference Engine API

## Class Overview

```cpp
#include <tiny_llm/inference_engine.h>

namespace tiny_llm {

class InferenceEngine {
public:
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& model_path,
        const ModelConfig& config);

    Result<std::vector<int>> generate(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& config);

    const GenerationStats& getStats() const;
    void resetStats();

    static int sampleGreedy(const half* logits, int vocab_size);
    static int sampleTemperature(const half* logits, int vocab_size, float temperature,
                                 unsigned seed = 0);
    static int sampleTopK(const half* logits, int vocab_size, int k, float temperature,
                          unsigned seed = 0);
    static int sampleTopP(const half* logits, int vocab_size, float p, float temperature,
                          unsigned seed = 0);
};

} // namespace tiny_llm
```

## Runtime Loading

- `load("model.bin", config)` targets the supported binary runtime format.
- Passing a `.gguf` path returns a structured error explaining that direct GGUF runtime loading is not currently supported.
- GGUF parsing/inspection lives on `GGUFParser`.

## Usage Example

```cpp
#include <iostream>
#include <tiny_llm/inference_engine.h>

int main() {
    using namespace tiny_llm;

    ModelConfig config;
    auto engine_result = InferenceEngine::load("model.bin", config);
    if (engine_result.isErr()) {
        std::cerr << engine_result.error() << '\n';
        return 1;
    }

    auto engine = std::move(engine_result.value());

    GenerationConfig generation;
    generation.max_new_tokens = 64;
    generation.temperature = 0.7f;
    generation.top_p = 0.9f;
    generation.do_sample = true;

    auto output = engine->generate({1, 15043, 29892}, generation);
    if (output.isErr()) {
        std::cerr << output.error() << '\n';
        return 1;
    }

    const auto& stats = engine->getStats();
    std::cout << "Generated " << output.value().size()
              << " tokens at " << stats.tokens_per_second << " tok/s\n";
}
```

## Input and Output Contract

| Method | Input | Output |
|--------|-------|--------|
| `load()` | runtime model path + `ModelConfig` | `Result<std::unique_ptr<InferenceEngine>>` |
| `generate()` | prompt token IDs + `GenerationConfig` | `Result<std::vector<int>>` |
| `getStats()` | none | `GenerationStats` |

## Notes

- Prompts are token IDs, not raw text strings.
- Check `isErr()` before calling `value()` or `error()`.
