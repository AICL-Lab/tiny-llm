# Changelog

All notable tracked releases of Tiny-LLM are recorded here.

## Unreleased

### Added

- gpt2 风格字节级 BPE tokenizer：从 GGUF 读取 tokens/merges/token_type，
  手写 Qwen2 预分词正则（Unicode 感知）+ GPT-2 字节编码 + BPE 合并，
  支持 CONTROL/USER_DEFINED 特殊 token 精确隔离与字节级无损 decode
- `loadTokenizerData`：从 GGUF 元数据提取 TokenizerData
- 测试：tokenizer 差分测试（对照 HuggingFace tokenizers 库，30 例 417 token
  逐 id 对齐 + decode 无损往返），门控于 TLLM_GGUF_TEST_MODEL

- Q5_0 / Q4_K / Q6_K GGUF 反量化（Q4_K_M 文件的实际量化类型）
- 架构感知的 GGUF 配置提取：按 general.architecture 前缀读取（qwen2/llama/...），
  vocab_size 从 tokenizer.ggml.tokens 数组长度派生
- `tiny_llm_demo --inspect model.gguf`：CPU-only 的 GGUF 配置/tensor 摘要
- 测试：合成块反量化单元测试（期望值来自 Python gguf 参考实现）；
  真实模型门控测试（TLLM_GGUF_TEST_MODEL）

### Fixed

- **Qwen2 attention bias 缺失（GPU 端到端乱码根因）**：加载并应用 attn_q/k/v.bias，
  补齐 Qwen2 系 q/k/v 投影的 bias 项；修复后输出与 llama.cpp 前 14 token 完全一致
- **共享层工作区（OOM 修复）**：中间激活缓冲改为所有层复用（LayerWorkspace），
  修复 24 层每层独立分配导致的显存爆炸（0.5B 模型在 6GB 卡无法加载）
- **attention O 投影非就地（未初始化内存/不确定输出）**：注意力输出改用独立 attn_buf，
  修复就地 matmul 输入被覆盖导致的数据竞争与不确定生成
- **lm_head 支持 FP16**：output 层不量化，保持 logits 精度（W8A16 作为后备）
- calculateSize 不再对未知量化类型按 FP16 估算（会导致静默错位读取），改为显式失败
- 移除断言旧行为（"GGUF 运行时加载不支持"）的过时测试，改为验证真实的加载错误路径

### Tests

- W8A16 大矩阵差分测试（M*N >= 4096 走 tiled 分支，与 reference 对齐）
- Attention GQA decode 与 CPU 参考逐元素对比（此前仅验证"不 crash/非零"）
- 真实模型权重量化往返测试（反量化 -> 转置 -> W8A16 量化 -> 重建误差受控）
- demo CLI 支持 `--prompt` / `--max-tokens` / `--show-tokens` / `--use-reference`（GPU 端到端生成入口）

### Verified

- tokenizer：C++ encode 与 HuggingFace tokenizers 权威实现逐 id 一致
  （151936 词表，含 CJK/emoji/缩写/空白/特殊 token 等 30 例）

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
