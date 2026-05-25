# Inference Engine

`InferenceEngine` is the public runtime entry point for Tiny-LLM. It owns loaded runtime weights, creates the KV cache, builds the Transformer layer stack, and drives prefill/decode generation.

## Public API

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
};

} // namespace tiny_llm
```

## Loading Boundary

- `InferenceEngine::load()` uses the supported **binary runtime format**.
- `.gguf` paths are rejected by the runtime path with a structured error.
- GGUF parsing and metadata extraction live on `GGUFParser`, not on the runtime fast path.

## Responsibilities

1. Validate `ModelConfig` before runtime setup.
2. Load runtime weights through `ModelLoader::loadBin()`.
3. Allocate the CUDA stream and reusable hidden/logit buffers.
4. Create `KVCacheManager` and the `TransformerLayer` stack.
5. Run prefill once, then decode token-by-token until EOS or the generation limit.

## Inputs and Outputs

- **Input:** prompt token IDs (`std::vector<int>`)
- **Output:** generated token IDs via `Result<std::vector<int>>`
- **Stats:** prefill time, decode time, throughput, and token counts

## Failure Model

The runtime surface returns `Result<T>` errors for invalid config, unsupported runtime inputs, allocation failures, and generation validation failures. Callers should check `isErr()` before reading `value()`.
