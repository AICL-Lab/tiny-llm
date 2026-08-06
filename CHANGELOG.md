# Changelog

All notable tracked releases of Tiny-LLM are recorded here.

## Unreleased

### Added

- Q5_0 / Q4_K / Q6_K GGUF 反量化（Q4_K_M 文件的实际量化类型）
- 架构感知的 GGUF 配置提取：按 general.architecture 前缀读取（qwen2/llama/...），
  vocab_size 从 tokenizer.ggml.tokens 数组长度派生
- `tiny_llm_demo --inspect model.gguf`：CPU-only 的 GGUF 配置/tensor 摘要
- 测试：合成块反量化单元测试（期望值来自 Python gguf 参考实现）；
  真实模型门控测试（TLLM_GGUF_TEST_MODEL）

### Fixed

- calculateSize 不再对未知量化类型按 FP16 估算（会导致静默错位读取），改为显式失败
- 移除断言旧行为（"GGUF 运行时加载不支持"）的过时测试，改为验证真实的加载错误路径

### Verified

- Qwen2.5-0.5B-Instruct Q4_K_M（GGUF v3，291 tensors，469MB）：
  配置提取与 Q5_0/Q4_K/Q6_K 首块反量化同 Python gguf 参考实现一致

## Releases

| Version | Date | Summary |
|---|---|---|
| v2.0.2 | 2026-04-27 | Code quality improvements, GGUF quantization utilities, and repository cleanup |
| v2.0.1 | 2026-04-16 | Bug-fix release for scale-dimension handling and loader cleanup |
| v2.0.0 | 2026-03-09 | Core engine milestone with KV cache API redesign |

## Policy

- This file is the only tracked changelog in the repository.
- Keep entries short and focused on meaningful user-facing or maintainer-relevant milestones.
- Do not duplicate release history in the documentation site.
