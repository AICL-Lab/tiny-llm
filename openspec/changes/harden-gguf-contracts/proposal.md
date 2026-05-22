## Why

Tiny-LLM's GGUF host-side path is currently too forgiving and too shallow. The parser accepts untrusted sizes without enough overflow protection, the loader validates required runtime tensors only after it starts allocating CUDA memory, and malformed metadata can influence `ModelConfig` in ways that do not reflect the GGUF schema. In practice this means crafted or incomplete GGUF files can trigger exceptions, oversized allocations, or fake-success paths instead of returning a clean `Result<T>` error.

## What Changes

- Harden GGUF parsing against integer-overflow and oversized-allocation paths.
- Make GGUF runtime loading validate required metadata and tensor presence before any CUDA allocation.
- Remove misleading fallback behavior in GGUF-to-`ModelConfig` mapping.
- Add regression tests for hostile/malformed GGUF inputs and incomplete runtime artifacts.
- Align the inference-engine capability delta with the stricter GGUF contract.

## Capabilities

### Modified Capabilities
- `inference-engine`: Tightens GGUF parsing and runtime-loading failure behavior so malformed or incomplete files fail explicitly instead of throwing or silently synthesizing invalid weights.

## Impact

- Affects `src/gguf_parser.cpp`, `src/model_loader.cpp`, and related tests.
- Adds stricter failure behavior for malformed/incomplete GGUF inputs.
- Improves alignment between `Result<T>`-based error handling and the actual GGUF host-side path.
