// 任务 4.2：第二个真实模型验证（不同 GQA 配置或 MQA）。
//
// 门控：设置 TLLM_GGUF_TEST_MODEL_2 指向第二个模型的 GGUF 文件后启用。
// 推荐模型（任选其一）：
//   - Llama-3.2-1B-Instruct（GQA 32→8，与 Qwen2.5-0.5B 的 14→2 差异明显）
//   - 任意 MQA 模型（num_kv_heads == 1）
//
// 验证内容：配置提取满足 GQA/MQA 不变量、与已验模型配置不同、端到端加载
// 并生成成功 —— 证明 attention 的 group_size = num_heads / num_kv_heads
// 映射不是只对一种配置正确。
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/inference_engine.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {

TEST(SecondModelTest, LoadsAndGeneratesWithDistinctGQA) {
    const char *path = std::getenv("TLLM_GGUF_TEST_MODEL_2");
    if (path == nullptr) {
        GTEST_SKIP() << "set TLLM_GGUF_TEST_MODEL_2 to a second GGUF "
                        "(different GQA ratio or MQA) to enable";
    }
    int         device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    tiny_llm::GGUFParser parser(path);
    auto                 parse_result = parser.parse();
    ASSERT_TRUE(parse_result.isOk()) << parse_result.error();
    auto config_result = parser.extractModelConfig();
    ASSERT_TRUE(config_result.isOk()) << config_result.error();
    const auto &config = config_result.value();

    // GQA/MQA 不变量
    EXPECT_GT(config.num_heads, 0);
    EXPECT_GT(config.num_kv_heads, 0);
    EXPECT_EQ(config.num_heads % config.num_kv_heads, 0)
        << "num_heads must be divisible by num_kv_heads";

    // 与已验模型（Qwen2.5-0.5B，14→2）不同的配置才构成“第二个验证”
    EXPECT_FALSE(config.num_heads == 14 && config.num_kv_heads == 2)
        << "TLLM_GGUF_TEST_MODEL_2 与已验模型配置相同（14→2），无法构成第二验证";

    // 端到端：加载并 greedy 生成（单 token prompt，避免依赖 tokenizer）
    auto engine = tiny_llm::InferenceEngine::load(path, config);
    ASSERT_TRUE(engine.isOk()) << engine.error();
    tiny_llm::GenerationConfig gen;
    gen.max_new_tokens = 8;
    gen.do_sample = false;
    auto tokens = engine.value()->generate({1}, gen);
    ASSERT_TRUE(tokens.isOk()) << tokens.error();
    EXPECT_FALSE(tokens.value().empty());
}

} // namespace
