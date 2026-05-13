# Quick Start

This tutorial will guide you through your first inference with Tiny-LLM.

## Prerequisites

Make sure you have [installed Tiny-LLM](/en/guide/installation) first.

## Basic Usage

### 1. Loading a Model

```cpp
#include <tiny-llm/inference_engine.h>

using namespace tinyllm;

int main() {
    // Create the inference engine
    InferenceEngine engine;

    // Load model from file
    Result<void> result = engine.load("model.bin");
    if (!result.ok()) {
        std::cerr << "Failed to load model: " << result.error() << std::endl;
        return 1;
    }

    return 0;
}
```

### 2. Running Inference

```cpp
#include <tiny-llm/inference_engine.h>
#include <iostream>

int main() {
    InferenceEngine engine;
    engine.load("model.bin");

    // Generate text from prompt
    std::string prompt = "The future of AI is";
    Result<std::string> result = engine.generate(prompt, /*max_tokens=*/50);

    if (result.ok()) {
        std::cout << "Generated: " << result.value() << std::endl;
    } else {
        std::cerr << "Error: " << result.error() << std::endl;
    }

    return 0;
}
```

### 3. Configuration Options

```cpp
GenerationConfig config;
config.max_tokens = 100;
config.temperature = 0.7f;
config.top_p = 0.9f;
config.top_k = 50;

Result<std::string> result = engine.generate(prompt, config);
```

## Building Your Application

```bash
# Compile your application
g++ -std=c++17 my_app.cpp -o my_app -I/path/to/tiny-llm/include -L/path/to/tiny-llm/build -ltinyllm
```

## Next Steps

- [Architecture Overview](/en/guide/architecture) - Learn about the internals
- [Configuration Guide](/en/guide/configuration) - Detailed configuration options
- [API Reference](/en/api/) - Complete API documentation
