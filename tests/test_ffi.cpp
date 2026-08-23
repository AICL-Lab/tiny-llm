// C ABI（ffi.h）端到端测试：load -> allocate_sequence -> step(prefill/decode)
// -> free_sequence -> free。
//
// 门控：设置 TLLM_GGUF_TEST_MODEL 指向真实 GGUF 模型后启用。
// 维度字段由 GGUF 提取，TinyLlmConfig 仅 max_batch_size 生效。
#include "rmsnorm.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/execution_common.h"
#include "tiny_llm/ffi.h"
#include "w8a16_matmul.cuh"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
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

    char           err[256] = {0};
    TinyLlmConfig  cfg = {0, 0, 0, 0, 0, 0, 16, 1, 0};
    TinyLlmHandle *h = tinyllm_load(path, &cfg, err, sizeof(err));
    ASSERT_NE(h, nullptr) << "load failed: " << err;

    ASSERT_EQ(tinyllm_allocate_sequence(h, 0, 64), 0);

    // prefill "Hello, how are you?"（token ids 由 llama.cpp tokenizer 给出）
    int           prompt[] = {9707, 11, 1246, 525, 498, 30};
    int           positions[] = {0, 1, 2, 3, 4, 5};
    int           seq_lens[] = {6};
    unsigned char pre[] = {1};
    int           seq_ids[] = {0};
    int           next = -1;
    float         logprobs[6] = {0.0f};
    ASSERT_EQ(tinyllm_step(h, seq_ids, prompt, positions, seq_lens, nullptr, nullptr, pre, 1, &next,
                           logprobs, 3),
              0);
    ASSERT_GE(next, 0);
    EXPECT_EQ(next, 358) << "Qwen2.5-0.5B 对 \"Hello, how are you?\" 的首生成 token 应为 358"
                            "（若模型文件变化需更新该期望）";
    // logprobs 格式：(token_id, logprob) 交错，logprob <= 0
    EXPECT_GT(logprobs[0], 0.0f); // token_id
    EXPECT_LE(logprobs[1], 0.0f); // logprob

    // decode 4 步
    int pos = 6;
    for (int i = 0; i < 4; ++i) {
        int           in = next;
        int           lens[] = {1};
        unsigned char dec[] = {0};
        int           sid[] = {0};
        ASSERT_EQ(
            tinyllm_step(h, sid, &in, &pos, lens, nullptr, nullptr, dec, 1, &next, nullptr, 0), 0);
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

TEST(FFITest, BatchedLogprobsBufferContract) {
    const char *path = model_path();
    if (path == nullptr) {
        GTEST_SKIP() << "set TLLM_GGUF_TEST_MODEL to a GGUF file to enable";
    }

    char           err[256] = {0};
    TinyLlmConfig  cfg = {0, 0, 0, 0, 0, 0, 16, 2, 0};
    TinyLlmHandle *h = tinyllm_load(path, &cfg, err, sizeof(err));
    ASSERT_NE(h, nullptr) << "load failed: " << err;
    ASSERT_EQ(tinyllm_allocate_sequence(h, 0, 64), 0);
    ASSERT_EQ(tinyllm_allocate_sequence(h, 1, 64), 0);

    int           seq_ids[] = {0, 1};
    int           prompt[] = {9707, 11, 1246, 525, 498, 30, 9707, 11, 1246, 525, 498, 30};
    int           positions[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
    int           seq_lens[] = {6, 6};
    unsigned char prefill[] = {1, 1};
    int           next_tokens[] = {-1, -1};

    EXPECT_EQ(tinyllm_step(h, seq_ids, prompt, positions, seq_lens, nullptr, nullptr, prefill, 2,
                           next_tokens, nullptr, -1),
              -1);
    EXPECT_EQ(tinyllm_step(h, seq_ids, prompt, positions, seq_lens, nullptr, nullptr, prefill, 2,
                           next_tokens, nullptr, 3),
              -1);

    constexpr int                    k = 3;
    constexpr float                  sentinel = 12345.0f;
    std::array<float, 2 * k * 2 + 2> storage;
    storage.fill(sentinel);
    float *logprobs = storage.data() + 1;
    ASSERT_EQ(tinyllm_step(h, seq_ids, prompt, positions, seq_lens, nullptr, nullptr, prefill, 2,
                           next_tokens, logprobs, k),
              0);

    EXPECT_EQ(storage.front(), sentinel);
    EXPECT_EQ(storage.back(), sentinel);
    for (int sequence = 0; sequence < 2; ++sequence) {
        EXPECT_GE(next_tokens[sequence], 0);
        for (int candidate = 0; candidate < k; ++candidate) {
            const int offset = (sequence * k + candidate) * 2;
            EXPECT_GE(logprobs[offset], 0.0f);
            EXPECT_LE(logprobs[offset + 1], 0.0f);
        }
    }

    EXPECT_EQ(tinyllm_free_sequence(h, 0), 0);
    EXPECT_EQ(tinyllm_free_sequence(h, 1), 0);
    tinyllm_free(h);
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
    tiny_llm::ModelWeights       weights;
    tiny_llm::DeviceBuffer<half> d_final_norm(hidden_dim);
    {
        std::vector<half> w(hidden_dim);
        for (auto &v : w)
            v = __float2half(1.0f);
        d_final_norm.copyFromHost(w.data(), hidden_dim);
    }
    weights.final_norm_weight = d_final_norm.data();

    tiny_llm::DeviceBuffer<half> d_lm_head(static_cast<size_t>(hidden_dim) * vocab_size);
    {
        std::vector<half> w(static_cast<size_t>(hidden_dim) * vocab_size);
        for (auto &v : w)
            v = __float2half(dist(gen));
        d_lm_head.copyFromHost(w.data(), w.size());
    }
    weights.lm_head_fp16 = d_lm_head.data();
    // lm_head（W8A16）保持默认无效，验证走 FP16 分支

    // 输入 hidden（helper 就地 rmsnorm，因此准备一份副本做两步参考）
    tiny_llm::DeviceBuffer<half> d_hidden(hidden_dim);
    tiny_llm::DeviceBuffer<half> d_hidden_copy(hidden_dim);
    {
        std::vector<half> h(hidden_dim);
        for (auto &v : h)
            v = __float2half(dist(gen));
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
    tiny_llm::ModelWeights weights_no_norm;
    weights_no_norm.lm_head_fp16 = d_lm_head.data();

    tiny_llm::DeviceBuffer<half> d_hidden2(hidden_dim);
    tiny_llm::DeviceBuffer<half> d_hidden2_copy(hidden_dim);
    {
        std::vector<half> h(hidden_dim);
        for (auto &v : h)
            v = __float2half(dist(gen));
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

// D2e：策略 1（分页 KV）与策略 2（连续 KV）逐 token 差分测试（门控真模型）。
// 两个模式用不同句柄各自 load，不能共用同一句柄。
TEST(FFITest, PagedKVStrategyMatchesContiguous) {
    const char *path = model_path();
    if (path == nullptr) {
        GTEST_SKIP() << "set TLLM_GGUF_TEST_MODEL to a GGUF file to enable";
    }

    char          err[256] = {0};
    int           prompt[] = {9707, 11, 1246, 525, 498, 30}; // "Hello, how are you?"
    int           positions[] = {0, 1, 2, 3, 4, 5};
    int           seq_lens[] = {6};
    unsigned char pre[] = {1};
    int           seq_ids[] = {0};
    int           next = -1;

    // ── 连续模式（策略 2）基线：prefill 6 token + decode 4 步 ──
    TinyLlmConfig  cfg2 = {0, 0, 0, 0, 0, 0, 16, 1, 0};
    TinyLlmHandle *h2 = tinyllm_load(path, &cfg2, err, sizeof(err));
    ASSERT_NE(h2, nullptr) << "load failed: " << err;
    ASSERT_EQ(tinyllm_allocate_sequence(h2, 0, 64), 0);

    std::vector<int> seq_contig;
    ASSERT_EQ(tinyllm_step(h2, seq_ids, prompt, positions, seq_lens, nullptr, nullptr, pre, 1,
                           &next, nullptr, 0),
              0);
    ASSERT_GE(next, 0);
    seq_contig.push_back(next);
    int pos = 6;
    for (int i = 0; i < 4; ++i) {
        int           in = next;
        int           lens[] = {1};
        unsigned char dec[] = {0};
        ASSERT_EQ(
            tinyllm_step(h2, seq_ids, &in, &pos, lens, nullptr, nullptr, dec, 1, &next, nullptr, 0),
            0);
        ASSERT_GE(next, 0);
        seq_contig.push_back(next);
        ++pos;
    }
    ASSERT_EQ(tinyllm_free_sequence(h2, 0), 0);
    tinyllm_free(h2);

    // ── 分页模式（策略 1）：block_size=16, max_num_blocks=8 ──
    // prompt=6 < block_size=16 → num_blocks=1, block_table={0}
    TinyLlmConfig  cfg1 = {0, 0, 0, 0, 0, 0, 16, 1, 8};
    TinyLlmHandle *h1 = tinyllm_load(path, &cfg1, err, sizeof(err));
    ASSERT_NE(h1, nullptr) << "load failed: " << err;
    ASSERT_EQ(tinyllm_allocate_sequence(h1, 0, 64), 0);

    std::vector<int> seq_paged;
    int              bt[] = {0};
    int              nb = 1;
    ASSERT_EQ(
        tinyllm_step(h1, seq_ids, prompt, positions, seq_lens, bt, &nb, pre, 1, &next, nullptr, 0),
        0);
    ASSERT_GE(next, 0);
    seq_paged.push_back(next);
    pos = 6;
    for (int i = 0; i < 4; ++i) {
        int           in = next;
        int           lens[] = {1};
        unsigned char dec[] = {0};
        ASSERT_EQ(tinyllm_step(h1, seq_ids, &in, &pos, lens, bt, &nb, dec, 1, &next, nullptr, 0),
                  0);
        ASSERT_GE(next, 0);
        seq_paged.push_back(next);
        ++pos;
    }
    ASSERT_EQ(tinyllm_free_sequence(h1, 0), 0);
    tinyllm_free(h1);

    // 逐 token 完全一致
    ASSERT_EQ(seq_contig.size(), seq_paged.size());
    for (size_t i = 0; i < seq_contig.size(); ++i) {
        EXPECT_EQ(seq_contig[i], seq_paged[i]) << "token " << i << " differs";
    }

    // ── 跨块用例：decode 到 position >= 16（nb 从 1 升到 2）──
    // prompt 6 token 后，第 i 次 decode 的绝对位置 = 6 + i；pos=16 起需要 2 块。
    const int        decode_steps = 12; // 覆盖 pos 6..17，含跨块位置 16/17
    std::vector<int> seq_contig_x, seq_paged_x;

    TinyLlmHandle *h2b = tinyllm_load(path, &cfg2, err, sizeof(err));
    ASSERT_NE(h2b, nullptr) << "load failed: " << err;
    ASSERT_EQ(tinyllm_allocate_sequence(h2b, 0, 128), 0);
    ASSERT_EQ(tinyllm_step(h2b, seq_ids, prompt, positions, seq_lens, nullptr, nullptr, pre, 1,
                           &next, nullptr, 0),
              0);
    seq_contig_x.push_back(next);
    pos = 6;
    for (int i = 0; i < decode_steps; ++i) {
        int           in = next;
        int           lens[] = {1};
        unsigned char dec[] = {0};
        ASSERT_EQ(tinyllm_step(h2b, seq_ids, &in, &pos, lens, nullptr, nullptr, dec, 1, &next,
                               nullptr, 0),
                  0);
        ASSERT_GE(next, 0);
        seq_contig_x.push_back(next);
        ++pos;
    }
    ASSERT_EQ(tinyllm_free_sequence(h2b, 0), 0);
    tinyllm_free(h2b);

    TinyLlmHandle *h1b = tinyllm_load(path, &cfg1, err, sizeof(err));
    ASSERT_NE(h1b, nullptr) << "load failed: " << err;
    ASSERT_EQ(tinyllm_allocate_sequence(h1b, 0, 128), 0);
    ASSERT_EQ(
        tinyllm_step(h1b, seq_ids, prompt, positions, seq_lens, bt, &nb, pre, 1, &next, nullptr, 0),
        0);
    seq_paged_x.push_back(next);
    int bt2[] = {0, 1};
    int nb2 = 2;
    pos = 6;
    for (int i = 0; i < decode_steps; ++i) {
        int           in = next;
        int           lens[] = {1};
        unsigned char dec[] = {0};
        const int     cur_pos = 6 + i;
        // pos >= 16 需要 2 块；否则 1 块
        if (cur_pos >= 16) {
            ASSERT_EQ(
                tinyllm_step(h1b, seq_ids, &in, &pos, lens, bt2, &nb2, dec, 1, &next, nullptr, 0),
                0);
        } else {
            ASSERT_EQ(
                tinyllm_step(h1b, seq_ids, &in, &pos, lens, bt, &nb, dec, 1, &next, nullptr, 0), 0);
        }
        ASSERT_GE(next, 0);
        seq_paged_x.push_back(next);
        ++pos;
    }
    ASSERT_EQ(tinyllm_free_sequence(h1b, 0), 0);
    tinyllm_free(h1b);

    // 跨块用例逐 token 完全一致
    ASSERT_EQ(seq_contig_x.size(), seq_paged_x.size());
    for (size_t i = 0; i < seq_contig_x.size(); ++i) {
        EXPECT_EQ(seq_contig_x[i], seq_paged_x[i]) << "cross-block token " << i << " differs";
    }
}
