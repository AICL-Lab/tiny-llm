## Context

The repository is in final hardening mode. For this phase, honest failure is better than partial runtime optimism. The current GGUF host-side path violates that principle in three ways:

1. integer sizes from GGUF metadata/tensor descriptors are trusted too long
2. runtime tensor validation happens after CUDA allocation starts
3. schema mapping mixes valid configuration keys with unrelated metadata fallbacks

These problems are tightly coupled and live behind one shallow seam: `GGUFParser` + `ModelLoader::loadGGUF`.

## Goals / Non-Goals

**Goals:**
- Fail malformed or oversized GGUF inputs with `Result<T>` errors, not host exceptions.
- Validate runtime-required tensors before any CUDA allocation.
- Keep the GGUF host-side seam smaller and more honest.
- Add regression coverage for hostile and incomplete GGUF fixtures.

**Non-Goals:**
- Full GGUF runtime support for every quantization format.
- Reworking CUDA kernels or broader inference-engine execution flow.
- Changing binary model loading behavior.

## Decisions

### 1. Add explicit overflow guards in GGUF size math

`GGUFTensorInfo::numElements()`, `GGUFTensorInfo::calculateSize()`, and GGUF array readers will reject sizes that overflow `size_t` or exceed safe allocation math.

Why:
- This converts malicious/untrusted metadata into structured parse errors.
- It keeps the parser as deep module: callers do not need to duplicate size-validation logic.

### 2. Validate GGUF runtime completeness before CUDA allocation

`ModelLoader::loadGGUF()` will build a required-tensor checklist from the extracted `ModelConfig` and return an error if required runtime tensors are missing or unsupported.

Why:
- Missing tensors are input-validation failures, not CUDA failures.
- This concentrates loader policy in one place and removes fake-success zero placeholders from the public seam.

### 3. Remove schema-invalid config fallback behavior

`GGUFParser::extractModelConfig()` will stop treating unrelated metadata like `general.architecture` as numeric dimension fallbacks.

Why:
- `general.architecture` is architecture identity, not `hidden_dim`.
- Keeping schema mapping strict improves locality and makes malformed GGUF behavior predictable.

## Risks / Trade-offs

- Some GGUF files that previously limped through with zero-filled placeholders will now fail early. This is intentional hardening.
- Tests must use host-side fixtures that fail before CUDA allocation when the environment lacks a working GPU/runtime.

## Verification Plan

1. Add failing host-side tests for:
   - tensor-size overflow guards
   - oversized GGUF arrays
   - invalid config fallback behavior
   - incomplete GGUF runtime tensors returning `Result` errors without throwing
2. Implement minimal production changes to satisfy those tests.
3. Run the focused test target plus strongest available repository verification commands.
