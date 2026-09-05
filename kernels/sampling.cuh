#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// 在 device 上计算多行 logits 的 greedy argmax。相同最大值取较小 token id，
// 与 InferenceEngine::sampleGreedy 的顺序扫描语义一致。
void greedy_argmax_batch(const half *logits, int batch_size, int vocab_size, int *tokens,
                         cudaStream_t stream = 0);

// 单行兼容包装。
void greedy_argmax(const half *logits, int vocab_size, int *token, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
