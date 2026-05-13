# API Reference

Welcome to the Tiny-LLM API reference documentation.

## Core Components

| Component | Description |
|-----------|-------------|
| [InferenceEngine](/en/api/inference-engine) | Main inference engine class |
| [ModelConfig](/en/api/model-config) | Model hyperparameters |
| [Result\<T\>](/en/api/result) | Error handling wrapper |
| [KVCacheManager](/en/api/kv-cache) | Key-value cache management |

## Quick Example

```cpp
#include <tiny-llm/inference_engine.h>
#include <tiny-llm/model_config.h>
#include <tiny-llm/result.h>

using namespace tinyllm;

int main() {
    // Create engine
    InferenceEngine engine;

    // Load model
    Result<void> load_result = engine.load("model.bin");
    if (!load_result.ok()) {
        std::cerr << "Error: " << load_result.error() << std::endl;
        return 1;
    }

    // Generate text
    GenerationConfig config;
    config.max_tokens = 100;
    config.temperature = 0.7f;

    Result<std::string> result = engine.generate("Hello, world!", config);
    if (result.ok()) {
        std::cout << result.value() << std::endl;
    }

    return 0;
}
```

## Error Handling

All operations return `Result\<T\>` for safe error propagation:

```cpp
Result\<T\> result = some_operation();
if (!result.ok()) {
    // Handle error
    std::cerr << result.error() << std::endl;
} else {
    // Use result
    T value = result.value();
}
```

## Namespaces

All public APIs are in the `tinyllm` namespace:

```cpp
using namespace tinyllm;
// or
tinyllm::InferenceEngine engine;
```

## Header Files

| Header | Description |
|--------|-------------|
| `<tiny-llm/inference_engine.h>` | Main engine class |
| `<tiny-llm/model_config.h>` | Configuration structs |
| `<tiny-llm/result.h>` | Result type |
| `<tiny-llm/kv_cache.h>` | Cache management |
| `<tiny-llm/tokenizer.h>` | Tokenization utilities |
