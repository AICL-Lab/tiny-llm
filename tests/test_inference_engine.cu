#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/inference_engine.h"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include <random>
// #include <rapidcheck.h>
// NOTE: rapidcheck/gtest is disabled in .cu tests due to GCC 11/12 + nvcc
// std::function compatibility issues during CI builds.
// #include <rapidcheck/gtest.h>
#include <set>
#include <vector>

using namespace tiny_llm;

// Helper class for InferenceEngine tests (CPU-based sampling tests)
class InferenceEngineTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Sampling tests are CPU-only, no GPU required
    }

    void TearDown() override {}

    // Generate random FP16 logits
    std::vector<half> randomLogits(int vocab_size, float scale = 10.0f, unsigned seed = 42) {
        std::vector<half>                     data(vocab_size);
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto &v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }

    // Find argmax of logits
    int argmax(const std::vector<half> &logits) {
        int   max_idx = 0;
        float max_val = __half2float(logits[0]);
        for (size_t i = 1; i < logits.size(); ++i) {
            float val = __half2float(logits[i]);
            if (val > max_val) {
                max_val = val;
                max_idx = static_cast<int>(i);
            }
        }
        return max_idx;
    }
};

// Unit test: Greedy sampling returns argmax
TEST_F(InferenceEngineTest, GGUFLoadFailsOnMissingFile) {
    // GGUF 运行时加载已实现；不存在的文件应返回解析错误
    ModelConfig config;
    auto        result = InferenceEngine::load("missing-model.gguf", config);
    EXPECT_TRUE(result.isErr());
    EXPECT_NE(result.error().find("Failed to parse GGUF"), std::string::npos);
}

TEST_F(InferenceEngineTest, GreedySamplingReturnsArgmax) {
    int  vocab_size = 1000;
    auto logits = randomLogits(vocab_size, 10.0f, 123);

    int expected = argmax(logits);
    int actual = InferenceEngine::sampleGreedy(logits.data(), vocab_size);

    EXPECT_EQ(actual, expected);
}

// Unit test: Greedy sampling with clear maximum
TEST_F(InferenceEngineTest, GreedySamplingClearMaximum) {
    int               vocab_size = 100;
    std::vector<half> logits(vocab_size, __float2half(0.0f));

    // Set a clear maximum at position 42
    logits[42] = __float2half(100.0f);

    int result = InferenceEngine::sampleGreedy(logits.data(), vocab_size);
    EXPECT_EQ(result, 42);
}

// Unit test: Temperature sampling produces valid indices
TEST_F(InferenceEngineTest, TemperatureSamplingValidIndex) {
    int  vocab_size = 1000;
    auto logits = randomLogits(vocab_size, 10.0f, 456);

    for (int i = 0; i < 100; ++i) {
        int result = InferenceEngine::sampleTemperature(logits.data(), vocab_size, 1.0f, i);
        EXPECT_GE(result, 0);
        EXPECT_LT(result, vocab_size);
    }
}

// Unit test: Top-k sampling produces valid indices
TEST_F(InferenceEngineTest, TopKSamplingValidIndex) {
    int  vocab_size = 1000;
    int  k = 50;
    auto logits = randomLogits(vocab_size, 10.0f, 789);

    for (int i = 0; i < 100; ++i) {
        int result = InferenceEngine::sampleTopK(logits.data(), vocab_size, k, 1.0f, i);
        EXPECT_GE(result, 0);
        EXPECT_LT(result, vocab_size);
    }
}

// Unit test: Top-p sampling produces valid indices
TEST_F(InferenceEngineTest, TopPSamplingValidIndex) {
    int   vocab_size = 1000;
    float p = 0.9f;
    auto  logits = randomLogits(vocab_size, 10.0f, 101);

    for (int i = 0; i < 100; ++i) {
        int result = InferenceEngine::sampleTopP(logits.data(), vocab_size, p, 1.0f, i);
        EXPECT_GE(result, 0);
        EXPECT_LT(result, vocab_size);
    }
}

// Unit test: Low temperature approaches greedy
TEST_F(InferenceEngineTest, LowTemperatureApproachesGreedy) {
    int  vocab_size = 100;
    auto logits = randomLogits(vocab_size, 10.0f, 202);

    int greedy_result = InferenceEngine::sampleGreedy(logits.data(), vocab_size);

    // With very low temperature, should almost always return argmax
    int matches = 0;
    for (int i = 0; i < 100; ++i) {
        int result = InferenceEngine::sampleTemperature(logits.data(), vocab_size, 0.01f, i);
        if (result == greedy_result) ++matches;
    }

    // Should match at least 95% of the time
    EXPECT_GE(matches, 95);
}

TEST_F(InferenceEngineTest, SamplingHandlesInvalidInputsSafely) {
    EXPECT_EQ(InferenceEngine::sampleGreedy(nullptr, 0), 0);
    EXPECT_EQ(InferenceEngine::sampleTemperature(nullptr, 0, 0.0f, 1), 0);

    auto logits = randomLogits(16, 10.0f, 303);
    int  top_k_result = InferenceEngine::sampleTopK(logits.data(), 16, 1000, 0.0f, 7);
    EXPECT_GE(top_k_result, 0);
    EXPECT_LT(top_k_result, 16);

    int top_p_result = InferenceEngine::sampleTopP(logits.data(), 16, 0.0f, 0.0f, 9);
    EXPECT_GE(top_p_result, 0);
    EXPECT_LT(top_p_result, 16);
}

TEST_F(InferenceEngineTest, TopKSamplingClampsToVocabularySize) {
    auto logits = randomLogits(8, 10.0f, 404);
    int  result = InferenceEngine::sampleTopK(logits.data(), 8, 64, 1.0f, 11);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 8);
}

TEST_F(InferenceEngineTest, TopPSamplingClampsProbabilityRange) {
    auto logits = randomLogits(8, 10.0f, 505);
    int  low_result = InferenceEngine::sampleTopP(logits.data(), 8, -1.0f, 1.0f, 13);
    int  high_result = InferenceEngine::sampleTopP(logits.data(), 8, 2.0f, 1.0f, 17);
    EXPECT_GE(low_result, 0);
    EXPECT_LT(low_result, 8);
    EXPECT_GE(high_result, 0);
    EXPECT_LT(high_result, 8);
}

TEST_F(InferenceEngineTest, TemperatureSamplingClampsNonPositiveTemperature) {
    auto logits = randomLogits(8, 10.0f, 606);
    int  result = InferenceEngine::sampleTemperature(logits.data(), 8, 0.0f, 19);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 8);
}

TEST_F(InferenceEngineTest, GreedySamplingHandlesSingleTokenVocabulary) {
    half logit = __float2half(1.0f);
    EXPECT_EQ(InferenceEngine::sampleGreedy(&logit, 1), 0);
}

TEST_F(InferenceEngineTest, TopKSamplingHandlesNonPositiveK) {
    auto logits = randomLogits(8, 10.0f, 707);
    int  result = InferenceEngine::sampleTopK(logits.data(), 8, 0, 1.0f, 23);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 8);
}

TEST_F(InferenceEngineTest, TopPSamplingHandlesTinyPositiveProbability) {
    auto logits = randomLogits(8, 10.0f, 808);
    int  result = InferenceEngine::sampleTopP(logits.data(), 8, 1e-8f, 1.0f, 29);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 8);
}

TEST_F(InferenceEngineTest, TemperatureSamplingHandlesTinyPositiveTemperature) {
    auto logits = randomLogits(8, 10.0f, 909);
    int  result = InferenceEngine::sampleTemperature(logits.data(), 8, 1e-8f, 31);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 8);
}

TEST_F(InferenceEngineTest, TopKSamplingHandlesNullInputSafely) {
    EXPECT_EQ(InferenceEngine::sampleTopK(nullptr, 0, 5, 1.0f, 37), 0);
}

TEST_F(InferenceEngineTest, TopPSamplingHandlesNullInputSafely) {
    EXPECT_EQ(InferenceEngine::sampleTopP(nullptr, 0, 0.9f, 1.0f, 41), 0);
}

TEST_F(InferenceEngineTest, SamplingDoesNotReturnOutOfRangeIndexAfterClamping) {
    auto logits = randomLogits(3, 10.0f, 1001);
    for (int seed = 0; seed < 20; ++seed) {
        int a = InferenceEngine::sampleTemperature(logits.data(), 3, -5.0f, seed);
        int b = InferenceEngine::sampleTopK(logits.data(), 3, 999, -5.0f, seed + 100);
        int c = InferenceEngine::sampleTopP(logits.data(), 3, 5.0f, -5.0f, seed + 200);
        EXPECT_GE(a, 0);
        EXPECT_LT(a, 3);
        EXPECT_GE(b, 0);
        EXPECT_LT(b, 3);
        EXPECT_GE(c, 0);
        EXPECT_LT(c, 3);
    }
}

TEST_F(InferenceEngineTest, SamplingHandlesNegativeVocabularyGracefully) {
    EXPECT_EQ(InferenceEngine::sampleGreedy(nullptr, -1), 0);
    EXPECT_EQ(InferenceEngine::sampleTemperature(nullptr, -1, 1.0f, 1), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesNegativeProbabilityGracefully) {
    auto logits = randomLogits(5, 10.0f, 1111);
    int  result = InferenceEngine::sampleTopP(logits.data(), 5, -10.0f, 1.0f, 43);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 5);
}

TEST_F(InferenceEngineTest, SamplingHandlesHugeTemperatureGracefully) {
    auto logits = randomLogits(5, 10.0f, 1212);
    int  result = InferenceEngine::sampleTemperature(logits.data(), 5, 1e6f, 47);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 5);
}

TEST_F(InferenceEngineTest, SamplingHandlesHugeTopPGracefully) {
    auto logits = randomLogits(5, 10.0f, 1313);
    int  result = InferenceEngine::sampleTopP(logits.data(), 5, 1e6f, 1.0f, 53);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 5);
}

TEST_F(InferenceEngineTest, SamplingHandlesHugeTopKGracefully) {
    auto logits = randomLogits(5, 10.0f, 1414);
    int  result = InferenceEngine::sampleTopK(logits.data(), 5, 1000000, 1.0f, 59);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 5);
}

TEST_F(InferenceEngineTest, SamplingHandlesZeroVocabularyGracefully) {
    EXPECT_EQ(InferenceEngine::sampleGreedy(nullptr, 0), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesNullInputAcrossStrategies) {
    EXPECT_EQ(InferenceEngine::sampleTemperature(nullptr, 0, 1.0f, 61), 0);
    EXPECT_EQ(InferenceEngine::sampleTopK(nullptr, 0, 1, 1.0f, 67), 0);
    EXPECT_EQ(InferenceEngine::sampleTopP(nullptr, 0, 0.5f, 1.0f, 71), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesDegenerateSingleElementCases) {
    half logit = __float2half(0.0f);
    EXPECT_EQ(InferenceEngine::sampleTemperature(&logit, 1, 0.0f, 73), 0);
    EXPECT_EQ(InferenceEngine::sampleTopK(&logit, 1, 0, 0.0f, 79), 0);
    EXPECT_EQ(InferenceEngine::sampleTopP(&logit, 1, 0.0f, 0.0f, 83), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesLargeKOnSmallVocabulary) {
    auto logits = randomLogits(2, 10.0f, 1515);
    int  result = InferenceEngine::sampleTopK(logits.data(), 2, 9999, 1.0f, 89);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 2);
}

TEST_F(InferenceEngineTest, SamplingHandlesExtremePBounds) {
    auto logits = randomLogits(2, 10.0f, 1616);
    int  low = InferenceEngine::sampleTopP(logits.data(), 2, -999.0f, 1.0f, 97);
    int  high = InferenceEngine::sampleTopP(logits.data(), 2, 999.0f, 1.0f, 101);
    EXPECT_GE(low, 0);
    EXPECT_LT(low, 2);
    EXPECT_GE(high, 0);
    EXPECT_LT(high, 2);
}

TEST_F(InferenceEngineTest, SamplingHandlesExtremeTemperatureBounds) {
    auto logits = randomLogits(2, 10.0f, 1717);
    int  low = InferenceEngine::sampleTemperature(logits.data(), 2, -999.0f, 103);
    int  high = InferenceEngine::sampleTemperature(logits.data(), 2, 999.0f, 107);
    EXPECT_GE(low, 0);
    EXPECT_LT(low, 2);
    EXPECT_GE(high, 0);
    EXPECT_LT(high, 2);
}

TEST_F(InferenceEngineTest, SamplingHandlesZeroAndNegativeStrategiesGracefully) {
    auto logits = randomLogits(4, 10.0f, 1818);
    EXPECT_GE(InferenceEngine::sampleTopK(logits.data(), 4, -5, 1.0f, 109), 0);
    EXPECT_GE(InferenceEngine::sampleTopP(logits.data(), 4, -5.0f, 1.0f, 113), 0);
}

TEST_F(InferenceEngineTest, SamplingOutputsRemainBounded) {
    auto logits = randomLogits(4, 10.0f, 1919);
    for (int seed = 0; seed < 10; ++seed) {
        int result = InferenceEngine::sampleTopK(logits.data(), 4, seed - 5, 0.0f, seed + 2000);
        EXPECT_GE(result, 0);
        EXPECT_LT(result, 4);
    }
}

TEST_F(InferenceEngineTest, SamplingHandlesNullGreedyInputSafely) {
    EXPECT_EQ(InferenceEngine::sampleGreedy(nullptr, 1), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesNullTemperatureInputSafely) {
    EXPECT_EQ(InferenceEngine::sampleTemperature(nullptr, 1, 1.0f, 211), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesNullTopKInputSafely) {
    EXPECT_EQ(InferenceEngine::sampleTopK(nullptr, 1, 1, 1.0f, 223), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesNullTopPInputSafely) {
    EXPECT_EQ(InferenceEngine::sampleTopP(nullptr, 1, 0.9f, 1.0f, 227), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesInvalidAllAroundGracefully) {
    EXPECT_EQ(InferenceEngine::sampleGreedy(nullptr, 0), 0);
    EXPECT_EQ(InferenceEngine::sampleTemperature(nullptr, 0, -1.0f, 229), 0);
    EXPECT_EQ(InferenceEngine::sampleTopK(nullptr, 0, -1, -1.0f, 233), 0);
    EXPECT_EQ(InferenceEngine::sampleTopP(nullptr, 0, -1.0f, -1.0f, 239), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesSingleValueAcrossStrategies) {
    half logit = __float2half(2.0f);
    EXPECT_EQ(InferenceEngine::sampleGreedy(&logit, 1), 0);
    EXPECT_EQ(InferenceEngine::sampleTemperature(&logit, 1, 1.0f, 241), 0);
    EXPECT_EQ(InferenceEngine::sampleTopK(&logit, 1, 1, 1.0f, 251), 0);
    EXPECT_EQ(InferenceEngine::sampleTopP(&logit, 1, 1.0f, 1.0f, 257), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesMinimalProbabilityCase) {
    auto logits = randomLogits(3, 10.0f, 2020);
    int  result = InferenceEngine::sampleTopP(logits.data(), 3, 1e-12f, 1.0f, 263);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 3);
}

TEST_F(InferenceEngineTest, SamplingHandlesMinimalTemperatureCase) {
    auto logits = randomLogits(3, 10.0f, 2121);
    int  result = InferenceEngine::sampleTemperature(logits.data(), 3, 1e-12f, 269);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 3);
}

TEST_F(InferenceEngineTest, SamplingHandlesZeroKGracefully) {
    auto logits = randomLogits(3, 10.0f, 2222);
    int  result = InferenceEngine::sampleTopK(logits.data(), 3, 0, 1.0f, 271);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 3);
}

TEST_F(InferenceEngineTest, SamplingHandlesNegativeKGracefully) {
    auto logits = randomLogits(3, 10.0f, 2323);
    int  result = InferenceEngine::sampleTopK(logits.data(), 3, -100, 1.0f, 277);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 3);
}

TEST_F(InferenceEngineTest, SamplingHandlesProbabilityAboveOneGracefully) {
    auto logits = randomLogits(3, 10.0f, 2424);
    int  result = InferenceEngine::sampleTopP(logits.data(), 3, 100.0f, 1.0f, 281);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 3);
}

TEST_F(InferenceEngineTest, SamplingHandlesProbabilityBelowZeroGracefully) {
    auto logits = randomLogits(3, 10.0f, 2525);
    int  result = InferenceEngine::sampleTopP(logits.data(), 3, -100.0f, 1.0f, 283);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 3);
}

TEST_F(InferenceEngineTest, SamplingHandlesVeryLargeTemperatureGracefully) {
    auto logits = randomLogits(3, 10.0f, 2626);
    int  result = InferenceEngine::sampleTemperature(logits.data(), 3, 1e9f, 293);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 3);
}

TEST_F(InferenceEngineTest, SamplingHandlesVeryNegativeTemperatureGracefully) {
    auto logits = randomLogits(3, 10.0f, 2727);
    int  result = InferenceEngine::sampleTemperature(logits.data(), 3, -1e9f, 307);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, 3);
}

TEST_F(InferenceEngineTest, SamplingHandlesNullWithPositiveVocabularyGracefully) {
    EXPECT_EQ(InferenceEngine::sampleGreedy(nullptr, 3), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesNullProbabilityStrategiesGracefully) {
    EXPECT_EQ(InferenceEngine::sampleTopK(nullptr, 3, 2, 1.0f, 311), 0);
    EXPECT_EQ(InferenceEngine::sampleTopP(nullptr, 3, 0.5f, 1.0f, 313), 0);
}

TEST_F(InferenceEngineTest, SamplingHandlesSmallVocabularyAcrossStrategies) {
    auto logits = randomLogits(1, 10.0f, 2828);
    EXPECT_EQ(InferenceEngine::sampleGreedy(logits.data(), 1), 0);
    EXPECT_EQ(InferenceEngine::sampleTemperature(logits.data(), 1, 1.0f, 317), 0);
}

// 任务 3.3：CUDA Graphs 开/关的 generate() 差分测试。
// 门控于 TLLM_GGUF_TEST_MODEL（与现有真实模型门控测试风格一致）。
// 同一模型分别以 TLLM_CUDA_GRAPHS=0/1 生成，必须逐 token 一致。
TEST_F(InferenceEngineTest, CudaGraphsGenerateMatchesNonGraph) {
    const char *path = std::getenv("TLLM_GGUF_TEST_MODEL");
    if (path == nullptr) {
        GTEST_SKIP() << "set TLLM_GGUF_TEST_MODEL to a GGUF file to enable";
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

    // 与 FFI 门控测试相同的 prompt（token ids 由 llama.cpp tokenizer 给出）
    const std::vector<int> prompt = {9707, 11, 1246, 525, 498, 30};
    tiny_llm::GenerationConfig gen_config;
    gen_config.max_new_tokens = 16;
    gen_config.do_sample = false; // greedy

    // 任务 C2：CUDA Graphs 默认开启；TLLM_CUDA_GRAPHS=0 显式关闭。
    // 1) graphs off（显式 TLLM_CUDA_GRAPHS=0）
    setenv("TLLM_CUDA_GRAPHS", "0", 1);
    auto engine_off = tiny_llm::InferenceEngine::load(path, config);
    unsetenv("TLLM_CUDA_GRAPHS");
    ASSERT_TRUE(engine_off.isOk()) << engine_off.error();
    auto tokens_off = engine_off.value()->generate(prompt, gen_config);
    ASSERT_TRUE(tokens_off.isOk()) << tokens_off.error();

    // 2) graphs on（默认即开启；显式设 1 以确保语义明确）
    setenv("TLLM_CUDA_GRAPHS", "1", 1);
    auto engine_on = tiny_llm::InferenceEngine::load(path, config);
    unsetenv("TLLM_CUDA_GRAPHS");
    ASSERT_TRUE(engine_on.isOk()) << engine_on.error();
    auto tokens_on = engine_on.value()->generate(prompt, gen_config);
    ASSERT_TRUE(tokens_on.isOk()) << tokens_on.error();

    ASSERT_EQ(tokens_off.value().size(), tokens_on.value().size())
        << "graph on/off generated different number of tokens";
    for (size_t i = 0; i < tokens_off.value().size(); ++i) {
        EXPECT_EQ(tokens_off.value()[i], tokens_on.value()[i])
            << "token " << i << " differs: non-graph=" << tokens_off.value()[i]
            << " graph=" << tokens_on.value()[i];
    }
}

// 任务 4.1：超长输入失败路径。prompt + max_new_tokens 超过 max_seq_len 时，
// generate() 必须在分配 KV 之前返回验证错误，活跃序列数不增加、不崩溃。
TEST_F(InferenceEngineTest, GenerateRejectsOverlongInput) {
    // 小型合成配置：1 层、max_seq_len=16
    ModelConfig config;
    config.vocab_size = 1000;
    config.hidden_dim = 64;
    config.num_layers = 1;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = 64;
    config.intermediate_dim = 128;
    config.max_seq_len = 16;
    config.rope_theta = 10000.0f;
    config.rms_norm_eps = 1e-5f;

    // 空权重即可：generate() 在校验阶段就返回，不会触碰权重。
    ModelWeights weights;
    weights.layers.resize(1);

    InferenceEngine engine(config, std::move(weights));
    EXPECT_EQ(engine.getActiveSequenceCount(), 0);

    GenerationConfig gen;
    gen.max_new_tokens = 8;
    gen.do_sample = false;

    // 20 个 token 的 prompt：20 + 8 > 16（max_seq_len）
    std::vector<int> prompt(20, 1);
    auto             result = engine.generate(prompt, gen);
    EXPECT_TRUE(result.isErr()) << "overlong input must be rejected before KV allocation";
    EXPECT_TRUE(result.error().find("length") != std::string::npos ||
                result.error().find("max_seq_len") != std::string::npos ||
                result.error().find("too") != std::string::npos ||
                result.error().find("exceed") != std::string::npos)
        << "unexpected error message: " << result.error();

    // 拒绝后不应分配 KV 序列
    EXPECT_EQ(engine.getActiveSequenceCount(), 0);
}

#if 0
// Property-based tests
// Feature: tiny-llm-inference-engine, Property 6: Greedy Sampling Correctness
// Validates: Requirements 5.2
// NOTE: Disabled in CUDA translation units due to GCC 11/12 + nvcc
// compatibility issues with rapidcheck's GTest integration.

class SamplingPropertyTest : public InferenceEngineTest {};

// Property 6: Greedy Sampling Correctness
// For any logits tensor, greedy sampling must return the index of the maximum
// value
RC_GTEST_FIXTURE_PROP(SamplingPropertyTest, GreedySamplingEqualsArgmax,
                      (int vocab_raw, unsigned seed)) {
    // Constrain vocab size to reasonable range
    int vocab_size = 100 + (std::abs(vocab_raw) % 9900); // 100 to 10000

    // Generate random logits
    std::vector<half>                     logits(vocab_size);
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (auto &v : logits) {
        v = __float2half(dist(gen));
    }

    // Find expected argmax
    int   expected = 0;
    float max_val = __half2float(logits[0]);
    for (int i = 1; i < vocab_size; ++i) {
        float val = __half2float(logits[i]);
        if (val > max_val) {
            max_val = val;
            expected = i;
        }
    }

    // Property: greedy sampling should return argmax
    int actual = InferenceEngine::sampleGreedy(logits.data(), vocab_size);
    RC_ASSERT(actual == expected);
}

// Property: Temperature sampling produces valid indices
RC_GTEST_FIXTURE_PROP(SamplingPropertyTest, TemperatureSamplingValidRange,
                      (int vocab_raw, float temp_raw, unsigned seed)) {
    int   vocab_size = 100 + (std::abs(vocab_raw) % 9900);
    float temperature = 0.1f + std::abs(temp_raw) * 0.01f; // 0.1 to ~inf
    temperature = std::min(temperature, 10.0f);            // Cap at 10

    std::vector<half>                     logits(vocab_size);
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : logits) {
        v = __float2half(dist(gen));
    }

    int result = InferenceEngine::sampleTemperature(logits.data(), vocab_size, temperature, seed);

    RC_ASSERT(result >= 0);
    RC_ASSERT(result < vocab_size);
}

// Property: Top-k sampling only selects from top-k tokens
RC_GTEST_FIXTURE_PROP(SamplingPropertyTest, TopKSamplingFromTopK,
                      (int vocab_raw, int k_raw, unsigned seed)) {
    int vocab_size = 100 + (std::abs(vocab_raw) % 900);        // 100 to 1000
    int k = 1 + (std::abs(k_raw) % std::min(vocab_size, 100)); // 1 to min(vocab_size, 100)

    std::vector<half>                     logits(vocab_size);
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : logits) {
        v = __float2half(dist(gen));
    }

    // Find top-k indices
    std::vector<std::pair<float, int>> sorted_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        sorted_logits[i] = {__half2float(logits[i]), i};
    }
    std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + k, sorted_logits.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

    std::set<int> top_k_indices;
    for (int i = 0; i < k; ++i) {
        top_k_indices.insert(sorted_logits[i].second);
    }

    // Sample multiple times and verify all results are in top-k
    for (int trial = 0; trial < 10; ++trial) {
        int result = InferenceEngine::sampleTopK(logits.data(), vocab_size, k, 1.0f, seed + trial);
        RC_ASSERT(top_k_indices.count(result) > 0);
    }
}

// Property: Top-p sampling produces valid indices
RC_GTEST_FIXTURE_PROP(SamplingPropertyTest, TopPSamplingValidRange,
                      (int vocab_raw, float p_raw, unsigned seed)) {
    int   vocab_size = 100 + (std::abs(vocab_raw) % 900);
    float p = 0.1f + std::abs(std::fmod(p_raw, 0.9f)); // 0.1 to 1.0
    p = std::min(p, 1.0f);

    std::vector<half>                     logits(vocab_size);
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : logits) {
        v = __float2half(dist(gen));
    }

    int result = InferenceEngine::sampleTopP(logits.data(), vocab_size, p, 1.0f, seed);

    RC_ASSERT(result >= 0);
    RC_ASSERT(result < vocab_size);
}
#endif

// 修复验证：repetition_penalty 逻辑（applyRepetitionPenalty，llama.cpp 语义）。
// 正 logit 除以 penalty、负 logit 乘以 penalty；penalty==1.0 或空历史为 no-op。
TEST(RepetitionPenaltyTest, AppliesLlamaSemantics) {
    // 正 logit：token 0 在历史中，penalty=2.0 → 1.0/2 = 0.5
    std::vector<half> logits = {__float2half(1.0f), __float2half(1.0f)};
    InferenceEngine::applyRepetitionPenalty(logits.data(), {0}, 2.0f, 2);
    EXPECT_NEAR(__half2float(logits[0]), 0.5f, 1e-6f);
    EXPECT_NEAR(__half2float(logits[1]), 1.0f, 1e-6f);

    // 负 logit：token 0 在历史中，penalty=2.0 → -1.0*2 = -2.0
    std::vector<half> neg = {__float2half(-1.0f), __float2half(-1.0f)};
    InferenceEngine::applyRepetitionPenalty(neg.data(), {0}, 2.0f, 2);
    EXPECT_NEAR(__half2float(neg[0]), -2.0f, 1e-6f);
    EXPECT_NEAR(__half2float(neg[1]), -1.0f, 1e-6f);

    // penalty == 1.0 → no-op
    std::vector<half> same = {__float2half(0.25f), __float2half(0.75f)};
    InferenceEngine::applyRepetitionPenalty(same.data(), {0}, 1.0f, 2);
    EXPECT_NEAR(__half2float(same[0]), 0.25f, 1e-6f);
    EXPECT_NEAR(__half2float(same[1]), 0.75f, 1e-6f);

    // 空历史 → no-op
    std::vector<half> none = {__float2half(0.25f), __float2half(0.75f)};
    InferenceEngine::applyRepetitionPenalty(none.data(), {}, 2.0f, 2);
    EXPECT_NEAR(__half2float(none[0]), 0.25f, 1e-6f);

    // 越界 token id 被忽略
    std::vector<half> oob = {__float2half(0.25f), __float2half(0.75f)};
    InferenceEngine::applyRepetitionPenalty(oob.data(), {5, -1, 100}, 2.0f, 2);
    EXPECT_NEAR(__half2float(oob[0]), 0.25f, 1e-6f);
    EXPECT_NEAR(__half2float(oob[1]), 0.75f, 1e-6f);
}

// 修复验证：惩罚后 greedy 选择会避开历史 token（功能端到端语义）。
TEST(RepetitionPenaltyTest, AffectsGreedySelection) {
    // 两个 token logit 相等，token 0 在历史中 → 惩罚后 token 1 胜出
    std::vector<half> logits = {__float2half(1.0f), __float2half(1.0f)};
    std::vector<half> penalized = logits;
    InferenceEngine::applyRepetitionPenalty(penalized.data(), {0}, 2.0f, 2);
    int pick = InferenceEngine::sampleGreedy(penalized.data(), 2);
    EXPECT_EQ(pick, 1) << "重复 token 应被惩罚降低概率";
}
