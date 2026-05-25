# API Reference

Tiny-LLM exposes a compact public C++ surface under the `tiny_llm` namespace.

## Primary Headers

| Header | Purpose |
|--------|---------|
| `<tiny_llm/inference_engine.h>` | Runtime loading and token generation |
| `<tiny_llm/model_loader.h>` | Host-side runtime model loading helpers |
| `<tiny_llm/gguf_parser.h>` | GGUF parsing, metadata extraction, tensor inspection |
| `<tiny_llm/kv_cache.h>` | KV cache allocation and sequence management |
| `<tiny_llm/result.h>` | `Result<T>` error propagation |
| `<tiny_llm/types.h>` | Shared config, weight, and stats types |

## Quick Example

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
    auto output = engine->generate({1, 15043, 29892}, GenerationConfig{});
    if (output.isErr()) {
        std::cerr << output.error() << '\n';
        return 1;
    }
}
```

## Loading Surface

- Use `InferenceEngine::load()` for the supported binary runtime format.
- Use `GGUFParser` when you need GGUF parsing or metadata access.
- Do not assume a tokenizer or text-string generation API exists in the public surface.

## Reference Pages

- [InferenceEngine](/en/api/inference-engine)
- [Result&lt;T&gt;](/en/api/result)
- [KVCacheManager](/en/api/kv-cache)
- [ModelConfig](/en/api/model-config)
