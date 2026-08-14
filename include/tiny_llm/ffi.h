// C ABI 契约：paged-infer（Rust）与 tiny-llm（C++/CUDA）之间的执行后端桥接。
//
// 契约定义与 paged-infer 的 `src/tiny_llm_ffi.rs` 保持逐字段一致：
//   - TinyLlmConfig 为 8 个 int 的 repr(C) 布局（Rust 侧有布局守卫测试）
//   - 步进式执行：每步处理一个 batch（prefill/decode 混合）
//   - KV 生命周期由本后端管理（策略 2：连续 KV，block_tables 忽略）
//
// 数据布局约定：
//   - seq_ids 显式给出每序列 id；input_tokens / positions 是扁平化数组
//     （seq_lens 描述每序列切分，与 seq_ids 对齐）
//   - logprobs_k == 0 不输出；否则 logprobs 为
//     num_sequences * logprobs_k 的 (token_id, logprob) 交错数组
#ifndef TINY_LLM_FFI_H
#define TINY_LLM_FFI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TinyLlmHandle TinyLlmHandle;

typedef struct TinyLlmConfig {
    int hidden_dim;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    int block_size;      // 分页块大小（策略 2 下忽略，保留对齐）
    int max_batch_size;  // 引擎同时管理的最大序列数
} TinyLlmConfig;

// 加载 GGUF 模型；成功返回句柄，失败返回 NULL 并写入 err_buf。
TinyLlmHandle *tinyllm_load(const char *model_path, const TinyLlmConfig *config,
                            char *err_buf, int err_buf_len);

// 单步执行一个 batch（prefill/decode 混合），逐序列输出下一 token 与 logprobs。
// seq_ids 显式给出每个序列的 id（由 tinyllm_allocate_sequence 分配），
// 支持任意 id 的序列混批（返回 0 成功，非 0 错误码）。
int tinyllm_step(TinyLlmHandle *handle, const int *seq_ids, const int *input_tokens,
                 const int *positions, const int *seq_lens, const int *block_tables,
                 const unsigned char *is_prefill, int num_sequences, int *next_tokens,
                 float *logprobs, int logprobs_k);

// KV 生命周期：分配/释放一个序列的缓存。
int tinyllm_allocate_sequence(TinyLlmHandle *handle, int seq_id, int num_tokens);
int tinyllm_free_sequence(TinyLlmHandle *handle, int seq_id);

// 释放句柄及其全部资源。
void tinyllm_free(TinyLlmHandle *handle);

#ifdef __cplusplus
}
#endif

#endif // TINY_LLM_FFI_H
