// C ABI（ffi.h）端到端测试：load -> allocate_sequence -> step(prefill/decode)
// -> free_sequence -> free。
//
// 门控：设置 TLLM_GGUF_TEST_MODEL 指向真实 GGUF 模型后启用。
// 维度字段由 GGUF 提取，TinyLlmConfig 仅 max_batch_size 生效。
#include "tiny_llm/ffi.h"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/execution_common.h"
#include "rmsnorm.cuh"
#include "w8a16_matmul.cuh"
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <vector>

namespace {

const char *model_path() {
    static const char *p = std::getenv("TLLM_GGUF_TEST_MODEL");
    return p;
}

} // namespace

TEST(FFITest, LoadAllocateStepFree) {
    const char *path = model_path();
    if (path == nullptr) {
        GTEST_SKIP() << "set TLLM_GGUF_TEST_MODEL to a GGUF file to enable";
    }

    char err[256] = {0};
    TinyLlmConfig cfg = {0, 0, 0, 0, 0, 0, 16, 1, 0};
    TinyLlmHandle *h = tinyllm_load(path, &cfg, err, sizeof(err));
    ASSERT_NE(h, nullptr) << "load failed: " << err;

    ASSERT_EQ(tinyllm_allocate_sequence(h, 0, 64), 0);

    // prefill "Hello, how are you?"（token ids 由 llama.cpp tokenizer 给出）
    int prompt[] = {9707, 11, 1246, 525, 498, 30};
    int positions[] = {0, 1, 2, 3, 4, 5};
    int seq_lens[] = {6};
    unsigned char pre[] = {1};
    int seq_ids[] = {0};
    int next = -1;
    float logprobs[6] = {0.0f};
    ASSERT_EQ(tinyllm_step(h, seq_ids, prompt, positions, seq_lens, nullptr, nullptr, pre, 1,
                           &next, logprobs, 3),
              0);
    ASSERT_GE(next, 0);
    EXPECT_EQ(next, 358) << "Qwen2.5-0.5B 对 \"Hello, how are you?\" 的首生成 token 应为 358"
                            "（若模型文件变化需更新该期望）";
    // logprobs 格式：(token_id, logprob) 交错，logprob <= 0
    EXPECT_GT(logprobs[0], 0.0f);  // token_id
    EXPECT_LE(logprobs[1], 0.0f);  // logprob

    // decode 4 步
    int pos = 6;
    for (int i = 0; i < 4; ++i) {
        int in = next;
        int lens[] = {1};
        unsigned char dec[] = {0};
        int sid[] = {0};
        ASSERT_EQ(tinyllm_step(h, sid, &in, &pos, lens, nullptr, nullptr, dec, 1, &next, nullptr,
                               0),
                  0);
        ASSERT_GE(next, 0);
        ++pos;
    }

    ASSERT_EQ(tinyllm_free_sequence(h, 0), 0);
    tinyllm_free(h);
}

TEST(FFITest, InvalidArgsReturnError) {
    ASSERT_EQ(tinyllm_load(nullptr, nullptr, nullptr, 0), nullptr);
    TinyLlmHandle *h = nullptr;
    ASSERT_EQ(tinyllm_step(h, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
                           nullptr, nullptr, 0),
              -1);
    ASSERT_EQ(tinyllm_allocate_sequence(h, 0, 0), -1);
    tinyllm_free(nullptr); // 不崩溃
}

// 任务 4.3：execution_common helper 最小测试。
// fake ModelWeights（final_norm 权重全 1、lm_head_fp16 小矩阵），与
// "先 rmsnorm 再 fp16_matmul" 的两步调用逐元素比较（容差 1e-2）；
// 并覆盖 final_norm_weight == nullptr 分支。
TEST(ExecutionCommonTest, FinalNormAndComputeLogitsMatchesTwoStep) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    const int   hidden_dim = 64;
    const int   vocab_size = 128;
    const float eps = 1e-5f;

    tiny_llm::ModelConfig config;
    config.hidden_dim = hidden_dim;
    config.vocab_size = vocab_size;
    config.rms_norm_eps = eps;

    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // fake 权重：final_norm 全 1 + lm_head_fp16 随机小矩阵
    tiny_llm::ModelWeights   weights;
    tiny_llm::DeviceBuffer<half> d_final_norm(hidden_dim);
    {
        std::vector<half> w(hidden_dim);
        for (auto &v : w) v = __float2half(1.0f);
        d_final_norm.copyFromHost(w.data(), hidden_dim);
    }
    weights.final_norm_weight = d_final_norm.data();

    tiny_llm::DeviceBuffer<half> d_lm_head(static_cast<size_t>(hidden_dim) * vocab_size);
    {
        std::vector<half> w(static_cast<size_t>(hidden_dim) * vocab_size);
        for (auto &v : w) v = __float2half(dist(gen));
        d_lm_head.copyFromHost(w.data(), w.size());
    }
    weights.lm_head_fp16 = d_lm_head.data();
    // lm_head（W8A16）保持默认无效，验证走 FP16 分支

    // 输入 hidden（helper 就地 rmsnorm，因此准备一份副本做两步参考）
    tiny_llm::DeviceBuffer<half> d_hidden(hidden_dim);
    tiny_llm::DeviceBuffer<half> d_hidden_copy(hidden_dim);
    {
        std::vector<half> h(hidden_dim);
        for (auto &v : h) v = __float2half(dist(gen));
        d_hidden.copyFromHost(h.data(), hidden_dim);
        d_hidden_copy.copyFromHost(h.data(), hidden_dim);
    }

    // 两步参考：rmsnorm -> fp16_matmul
    tiny_llm::DeviceBuffer<half> d_normed(hidden_dim);
    tiny_llm::DeviceBuffer<half> d_ref_logits(vocab_size);
    tiny_llm::kernels::rmsnorm(d_hidden_copy.data(), weights.final_norm_weight, d_normed.data(), 1,
                               hidden_dim, eps);
    tiny_llm::kernels::fp16_matmul(d_normed.data(), weights.lm_head_fp16, d_ref_logits.data(), 1,
                                   vocab_size, hidden_dim);

    // helper
    tiny_llm::DeviceBuffer<half> d_logits(vocab_size);
    tiny_llm::finalNormAndComputeLogits(d_hidden.data(), weights, config, d_logits.data());

    cudaDeviceSynchronize();

    std::vector<half> logits(vocab_size), ref_logits(vocab_size);
    d_logits.copyToHost(logits.data(), vocab_size);
    d_ref_logits.copyToHost(ref_logits.data(), vocab_size);
    cudaDeviceSynchronize();

    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(__half2float(logits[i]), __half2float(ref_logits[i]), 1e-2f)
            << "logits[" << i << "] differs";
    }

    // final_norm_weight == nullptr 分支：helper 应只做 lm_head
    tiny_llm::ModelWeights   weights_no_norm;
    weights_no_norm.lm_head_fp16 = d_lm_head.data();

    tiny_llm::DeviceBuffer<half> d_hidden2(hidden_dim);
    tiny_llm::DeviceBuffer<half> d_hidden2_copy(hidden_dim);
    {
        std::vector<half> h(hidden_dim);
        for (auto &v : h) v = __float2half(dist(gen));
        d_hidden2.copyFromHost(h.data(), hidden_dim);
        d_hidden2_copy.copyFromHost(h.data(), hidden_dim);
    }

    tiny_llm::DeviceBuffer<half> d_ref2(vocab_size);
    tiny_llm::kernels::fp16_matmul(d_hidden2_copy.data(), weights_no_norm.lm_head_fp16,
                                   d_ref2.data(), 1, vocab_size, hidden_dim);

    tiny_llm::DeviceBuffer<half> d_logits2(vocab_size);
    tiny_llm::finalNormAndComputeLogits(d_hidden2.data(), weights_no_norm, config,
                                        d_logits2.data());
    cudaDeviceSynchronize();

    std::vector<half> logits2(vocab_size), ref2(vocab_size);
    d_logits2.copyToHost(logits2.data(), vocab_size);
    d_ref2.copyToHost(ref2.data(), vocab_size);
    cudaDeviceSynchronize();

    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(__half2float(logits2[i]), __half2float(ref2[i]), 1e-2f)
            << "no-norm logits[" << i << "] differs";
    }
}
