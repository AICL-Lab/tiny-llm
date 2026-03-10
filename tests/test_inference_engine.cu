#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/inference_engine.h"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <set>
#include <vector>

using namespace tiny_llm;

// Helper class for InferenceEngine tests
class InferenceEngineTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    cudaSetDevice(0);
  }

  void TearDown() override { cudaDeviceSynchronize(); }

  // Generate random FP16 logits
  std::vector<half> randomLogits(int vocab_size, float scale = 10.0f,
                                 unsigned seed = 42) {
    std::vector<half> data(vocab_size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto &v : data) {
      v = __float2half(dist(gen));
    }
    return data;
  }

  // Find argmax of logits
  int argmax(const std::vector<half> &logits) {
    int max_idx = 0;
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
TEST_F(InferenceEngineTest, GreedySamplingReturnsArgmax) {
  int vocab_size = 1000;
  auto logits = randomLogits(vocab_size, 10.0f, 123);

  int expected = argmax(logits);
  int actual = InferenceEngine::sampleGreedy(logits.data(), vocab_size);

  EXPECT_EQ(actual, expected);
}

// Unit test: Greedy sampling with clear maximum
TEST_F(InferenceEngineTest, GreedySamplingClearMaximum) {
  int vocab_size = 100;
  std::vector<half> logits(vocab_size, __float2half(0.0f));

  // Set a clear maximum at position 42
  logits[42] = __float2half(100.0f);

  int result = InferenceEngine::sampleGreedy(logits.data(), vocab_size);
  EXPECT_EQ(result, 42);
}

// Unit test: Temperature sampling produces valid indices
TEST_F(InferenceEngineTest, TemperatureSamplingValidIndex) {
  int vocab_size = 1000;
  auto logits = randomLogits(vocab_size, 10.0f, 456);

  for (int i = 0; i < 100; ++i) {
    int result =
        InferenceEngine::sampleTemperature(logits.data(), vocab_size, 1.0f, i);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, vocab_size);
  }
}

// Unit test: Top-k sampling produces valid indices
TEST_F(InferenceEngineTest, TopKSamplingValidIndex) {
  int vocab_size = 1000;
  int k = 50;
  auto logits = randomLogits(vocab_size, 10.0f, 789);

  for (int i = 0; i < 100; ++i) {
    int result =
        InferenceEngine::sampleTopK(logits.data(), vocab_size, k, 1.0f, i);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, vocab_size);
  }
}

// Unit test: Top-p sampling produces valid indices
TEST_F(InferenceEngineTest, TopPSamplingValidIndex) {
  int vocab_size = 1000;
  float p = 0.9f;
  auto logits = randomLogits(vocab_size, 10.0f, 101);

  for (int i = 0; i < 100; ++i) {
    int result =
        InferenceEngine::sampleTopP(logits.data(), vocab_size, p, 1.0f, i);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, vocab_size);
  }
}

// Unit test: Low temperature approaches greedy
TEST_F(InferenceEngineTest, LowTemperatureApproachesGreedy) {
  int vocab_size = 100;
  auto logits = randomLogits(vocab_size, 10.0f, 202);

  int greedy_result = InferenceEngine::sampleGreedy(logits.data(), vocab_size);

  // With very low temperature, should almost always return argmax
  int matches = 0;
  for (int i = 0; i < 100; ++i) {
    int result =
        InferenceEngine::sampleTemperature(logits.data(), vocab_size, 0.01f, i);
    if (result == greedy_result)
      ++matches;
  }

  // Should match at least 95% of the time
  EXPECT_GE(matches, 95);
}

// Property-based tests
// Feature: tiny-llm-inference-engine, Property 6: Greedy Sampling Correctness
// Validates: Requirements 5.2

class SamplingPropertyTest : public InferenceEngineTest {};

// Property 6: Greedy Sampling Correctness
// For any logits tensor, greedy sampling must return the index of the maximum
// value
RC_GTEST_FIXTURE_PROP(SamplingPropertyTest, GreedySamplingEqualsArgmax,
                      (int vocab_raw, unsigned seed)) {
  // Constrain vocab size to reasonable range
  int vocab_size = 100 + (std::abs(vocab_raw) % 9900); // 100 to 10000

  // Generate random logits
  std::vector<half> logits(vocab_size);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  for (auto &v : logits) {
    v = __float2half(dist(gen));
  }

  // Find expected argmax
  int expected = 0;
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
  int vocab_size = 100 + (std::abs(vocab_raw) % 9900);
  float temperature = 0.1f + std::abs(temp_raw) * 0.01f; // 0.1 to ~inf
  temperature = std::min(temperature, 10.0f);            // Cap at 10

  std::vector<half> logits(vocab_size);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (auto &v : logits) {
    v = __float2half(dist(gen));
  }

  int result = InferenceEngine::sampleTemperature(logits.data(), vocab_size,
                                                  temperature, seed);

  RC_ASSERT(result >= 0);
  RC_ASSERT(result < vocab_size);
}

// Property: Top-k sampling only selects from top-k tokens
RC_GTEST_FIXTURE_PROP(SamplingPropertyTest, TopKSamplingFromTopK,
                      (int vocab_raw, int k_raw, unsigned seed)) {
  int vocab_size = 100 + (std::abs(vocab_raw) % 900); // 100 to 1000
  int k = 1 + (std::abs(k_raw) %
               std::min(vocab_size, 100)); // 1 to min(vocab_size, 100)

  std::vector<half> logits(vocab_size);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (auto &v : logits) {
    v = __float2half(dist(gen));
  }

  // Find top-k indices
  std::vector<std::pair<float, int>> sorted_logits(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    sorted_logits[i] = {__half2float(logits[i]), i};
  }
  std::partial_sort(
      sorted_logits.begin(), sorted_logits.begin() + k, sorted_logits.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  std::set<int> top_k_indices;
  for (int i = 0; i < k; ++i) {
    top_k_indices.insert(sorted_logits[i].second);
  }

  // Sample multiple times and verify all results are in top-k
  for (int trial = 0; trial < 10; ++trial) {
    int result = InferenceEngine::sampleTopK(logits.data(), vocab_size, k, 1.0f,
                                             seed + trial);
    RC_ASSERT(top_k_indices.count(result) > 0);
  }
}

// Property: Top-p sampling produces valid indices
RC_GTEST_FIXTURE_PROP(SamplingPropertyTest, TopPSamplingValidRange,
                      (int vocab_raw, float p_raw, unsigned seed)) {
  int vocab_size = 100 + (std::abs(vocab_raw) % 900);
  float p = 0.1f + std::abs(std::fmod(p_raw, 0.9f)); // 0.1 to 1.0
  p = std::min(p, 1.0f);

  std::vector<half> logits(vocab_size);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (auto &v : logits) {
    v = __float2half(dist(gen));
  }

  int result =
      InferenceEngine::sampleTopP(logits.data(), vocab_size, p, 1.0f, seed);

  RC_ASSERT(result >= 0);
  RC_ASSERT(result < vocab_size);
}
