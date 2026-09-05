#include "tiny_llm/execution_common.h"
#include "rmsnorm.cuh"
#include "w8a16_matmul.cuh"
#include <cstddef>

namespace tiny_llm {

void finalNormAndComputeLogitsBatch(half *hidden, int num_tokens, const ModelWeights &weights,
                                    const ModelConfig &config, half *logits, cudaStream_t stream) {
    if (num_tokens <= 0) return;

    // final norm（就地，与 InferenceEngine::finalNorm 一致）
    if (weights.final_norm_weight) {
        kernels::rmsnorm_inplace(hidden, weights.final_norm_weight, num_tokens, config.hidden_dim,
                                 config.rms_norm_eps, stream);
    }

    // LM head projection: hidden @ lm_head.T
    // 优先 FP16 lm_head（output 层不量化，保持 logits 精度与 llama.cpp 对齐）；
    // W8A16 版本作为后备。
    if (weights.lm_head_fp16) {
        kernels::fp16_matmul(hidden, weights.lm_head_fp16, weights.lm_head_fp16_t, logits,
                             num_tokens, config.vocab_size, config.hidden_dim, stream);
    } else if (weights.lm_head.isValid()) {
        kernels::w8a16_matmul(hidden, weights.lm_head.data, weights.lm_head.scales,
                              weights.lm_head.data_t, weights.lm_head.scales_t, logits, num_tokens,
                              config.vocab_size, config.hidden_dim, weights.lm_head.group_size,
                              stream);
    }
}

void finalNormAndComputeLogits(half *hidden, const ModelWeights &weights, const ModelConfig &config,
                               half *logits, cudaStream_t stream) {
    finalNormAndComputeLogitsBatch(hidden, 1, weights, config, logits, stream);
}

} // namespace tiny_llm
