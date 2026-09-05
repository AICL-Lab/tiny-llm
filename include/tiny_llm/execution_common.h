#pragma once

#include "tiny_llm/types.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {

// InferenceEngine 与 FFI（ffi.cpp）共用的“最终层”执行 helper，
// 避免两条执行路径的 rmsnorm + lm_head 逻辑漂移。
//
// 语义：
//   hidden 为 [num_tokens, hidden_dim] 的 device 内存，就地做 final RMSNorm
//   （若 final_norm_weight 非空），再投影 lm_head（优先 FP16 lm_head，W8A16 后备），
//   logits 写入 [num_tokens, vocab_size] 的 device 缓冲。本函数不负责采样。
void finalNormAndComputeLogitsBatch(half *hidden, int num_tokens, const ModelWeights &weights,
                                    const ModelConfig &config, half *logits,
                                    cudaStream_t stream = 0);

// 单 token 兼容包装，供 InferenceEngine 和请求 logprobs 的 FFI 路径使用。
void finalNormAndComputeLogits(half *hidden, const ModelWeights &weights, const ModelConfig &config,
                               half *logits, cudaStream_t stream = 0);

} // namespace tiny_llm
