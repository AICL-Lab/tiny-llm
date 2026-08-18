#pragma once

#include "tiny_llm/types.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {

// InferenceEngine 与 FFI（ffi.cpp）共用的“最终层”执行 helper，
// 避免两条执行路径的 rmsnorm + lm_head 逻辑漂移。
//
// 语义：
//   hidden 为单个 token 的隐藏状态（[hidden_dim] device 内存），就地做
//   final RMSNorm（若 final_norm_weight 非空），再投影 lm_head
//   （优先 FP16 lm_head，W8A16 后备），logits 写入 [vocab_size] 的
//   device 缓冲。本函数不负责采样。
//
// 注意：只支持 num_tokens == 1（单 token），不处理 batch。
void finalNormAndComputeLogits(half *hidden, const ModelWeights &weights,
                               const ModelConfig &config, half *logits,
                               cudaStream_t stream = 0);

} // namespace tiny_llm
