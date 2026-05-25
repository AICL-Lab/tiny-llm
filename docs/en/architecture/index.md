# Architecture Overview

Tiny-LLM centers on a narrow CUDA/C++ runtime: a supported binary runtime loader, explicit KV cache management, W8A16 quantized layers, and host-side `Result<T>` error propagation.

## System Sketch

```mermaid
flowchart LR
    A[Binary runtime model] --> B[InferenceEngine::load()]
    C[GGUF file] --> D[GGUFParser]
    B --> E[ModelWeights]
    E --> F[Transformer Layers]
    F --> G[KVCacheManager]
    F --> H[W8A16 Kernels]
    H --> I[Sampling]
    I --> J[Generated token IDs]
    D --> K[Metadata / Tensor Inspection]
```

## Core Components

| Component | Responsibility |
|-----------|----------------|
| `InferenceEngine` | Public runtime entry point for loading supported binary models and generating token IDs |
| `ModelLoader` | Host-side runtime weight loading for the supported binary format |
| `GGUFParser` | GGUF parsing, metadata extraction, and tensor inspection |
| `TransformerLayer` | W8A16 attention + FFN execution |
| `KVCacheManager` | Pre-allocated cache slots and sequence lifecycle |
| `Result<T>` | Fallible host-side API boundary |

## Loading Boundary

- **Runtime loading:** `InferenceEngine::load("model.bin", config)`
- **GGUF tooling:** `GGUFParser` for parse/inspect/validate workflows
- **Not supported:** direct `.gguf` runtime loading through `InferenceEngine::load()`

## Generation Flow

1. Load weights through the supported binary runtime path.
2. Feed prompt token IDs into `generate()`.
3. Run prefill across the Transformer stack.
4. Reuse KV cache state during decode.
5. Sample the next token ID from logits until EOS or `max_new_tokens`.

## Notes

- Public examples should use token IDs, not an in-repo tokenizer API.
- Public docs should describe the binary runtime path and GGUF parser separately.

## Next Steps

- [Inference Engine](/en/architecture/inference-engine)
- [API Reference](/en/api/)
- [Performance Guide](/en/performance/)
