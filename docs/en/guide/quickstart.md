# Quick Start

This page shows the smallest end-to-end runtime path that matches the current codebase.

## 1. Load a supported runtime model

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
}
```

## 2. Generate token IDs

```cpp
using namespace tiny_llm;

auto engine = std::move(engine_result.value());

GenerationConfig generation;
generation.max_new_tokens = 32;
generation.do_sample = true;
generation.temperature = 0.7f;

auto output = engine->generate({1, 15043, 29892}, generation);
if (output.isErr()) {
    std::cerr << output.error() << '\n';
    return 1;
}
```

## 3. Know the boundary

- `InferenceEngine::load()` uses the supported binary runtime format.
- `GGUFParser` is the entry point for GGUF parsing and inspection.
- The public runtime API works on token IDs, not raw strings.

## Next Steps

- [Architecture Overview](/en/architecture/)
- [API Reference](/en/api/)
- [Installation](/en/guide/installation)
