## 1. GGUF parser overflow hardening

- [ ] 1.1 Add failing tests for tensor-size overflow and oversized GGUF arrays
- [ ] 1.2 Guard `numElements()`, `calculateSize()`, and array byte-count math against overflow
- [ ] 1.3 Re-run the focused GGUF/parser tests

## 2. GGUF config and runtime contract hardening

- [ ] 2.1 Add failing tests for invalid metadata fallback and incomplete runtime tensor sets
- [ ] 2.2 Make `extractModelConfig()` ignore unrelated metadata fallbacks
- [ ] 2.3 Make `ModelLoader::loadGGUF()` validate required tensors before CUDA allocation and fail with `Result` errors
- [ ] 2.4 Re-run the focused GGUF/model-loader tests

## 3. Spec and verification alignment

- [ ] 3.1 Update the inference-engine change delta to describe the stricter GGUF contract
- [ ] 3.2 Run repository verification commands or strongest available substitutes and record environment limits
