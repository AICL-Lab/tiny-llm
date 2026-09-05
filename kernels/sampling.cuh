#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// 在 device 上计算单行 logits 的 greedy argmax。相同最大值取较小 token id，
// 与 InferenceEngine::sampleGreedy 的顺序扫描语义一致。
void greedy_argmax(const half *logits, int vocab_size, int *token, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
