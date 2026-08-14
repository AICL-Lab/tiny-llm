// C ABI（ffi.h）端到端测试：load -> allocate_sequence -> step(prefill/decode)
// -> free_sequence -> free。
//
// 门控：设置 TLLM_GGUF_TEST_MODEL 指向真实 GGUF 模型后启用。
// 维度字段由 GGUF 提取，TinyLlmConfig 仅 max_batch_size 生效。
#include "tiny_llm/ffi.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdio>

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
    TinyLlmConfig cfg = {0, 0, 0, 0, 0, 0, 16, 1};
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
    ASSERT_EQ(tinyllm_step(h, seq_ids, prompt, positions, seq_lens, nullptr, pre, 1, &next,
                           logprobs, 3),
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
        ASSERT_EQ(tinyllm_step(h, sid, &in, &pos, lens, nullptr, dec, 1, &next, nullptr, 0), 0);
        ASSERT_GE(next, 0);
        ++pos;
    }

    ASSERT_EQ(tinyllm_free_sequence(h, 0), 0);
    tinyllm_free(h);
}

TEST(FFITest, InvalidArgsReturnError) {
    ASSERT_EQ(tinyllm_load(nullptr, nullptr, nullptr, 0), nullptr);
    TinyLlmHandle *h = nullptr;
    ASSERT_EQ(tinyllm_step(h, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr,
                          nullptr, 0),
              -1);
    ASSERT_EQ(tinyllm_allocate_sequence(h, 0, 0), -1);
    tinyllm_free(nullptr); // 不崩溃
}
