#include "tiny_llm/cuda_streams.h"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/inference_engine.h"
#include "tiny_llm/model_loader.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <random>
#include <vector>

using namespace tiny_llm;

// Helper class for integration tests
class IntegrationTest : public ::testing::Test {
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

  // Create a tiny test model config
  ModelConfig createTinyConfig() {
    ModelConfig config;
    config.vocab_size = 256;
    config.hidden_dim = 64;
    config.num_layers = 2;
    config.num_heads = 4;
    config.num_kv_heads = 4;
    config.head_dim = 16;
    config.intermediate_dim = 128;
    config.max_seq_len = 64;
    config.rope_theta = 10000.0f;
    config.rms_norm_eps = 1e-5f;
    config.eos_token_id = 2;
    config.bos_token_id = 1;
    return config;
  }

  // Generate random FP16 tensor on device
  half *randomDeviceFP16(int size, float scale = 1.0f, unsigned seed = 42) {
    std::vector<half> h_data(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto &v : h_data) {
      v = __float2half(dist(gen));
    }

    half *d_data;
    cudaMalloc(&d_data, size * sizeof(half));
    cudaMemcpy(d_data, h_data.data(), size * sizeof(half),
               cudaMemcpyHostToDevice);
    return d_data;
  }

  // Generate random INT8 tensor on device
  int8_t *randomDeviceINT8(int size, unsigned seed = 123) {
    std::vector<int8_t> h_data(size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto &v : h_data) {
      v = static_cast<int8_t>(dist(gen));
    }

    int8_t *d_data;
    cudaMalloc(&d_data, size * sizeof(int8_t));
    cudaMemcpy(d_data, h_data.data(), size * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    return d_data;
  }

  // Create random quantized weight
  QuantizedWeight createRandomWeight(int rows, int cols, int group_size,
                                     unsigned seed) {
    QuantizedWeight w;
    w.rows = rows;
    w.cols = cols;
    w.group_size = group_size;
    w.data = randomDeviceINT8(rows * cols, seed);

    int num_groups = (cols + group_size - 1) / group_size;
    w.scales = randomDeviceFP16(rows * num_groups, 0.1f, seed + 1000);

    return w;
  }

  // Free quantized weight
  void freeWeight(QuantizedWeight &w) {
    if (w.data) {
      cudaFree(w.data);
      w.data = nullptr;
    }
    if (w.scales) {
      cudaFree(w.scales);
      w.scales = nullptr;
    }
  }
};

// Test: Stream pool creation and usage
TEST_F(IntegrationTest, StreamPoolBasicUsage) {
  StreamPool pool(4);

  EXPECT_EQ(pool.numStreams(), 4);

  // Get streams in round-robin
  cudaStream_t s0 = pool.getStream();
  cudaStream_t s1 = pool.getStream();
  cudaStream_t s2 = pool.getStream();
  cudaStream_t s3 = pool.getStream();
  cudaStream_t s4 = pool.getStream(); // Should wrap around

  EXPECT_NE(s0, nullptr);
  EXPECT_NE(s1, nullptr);
  EXPECT_EQ(s0, s4); // Round-robin wrap

  pool.synchronizeAll();
}

// Test: CUDA event timing
TEST_F(IntegrationTest, CudaEventTiming) {
  CudaEvent start, end;

  start.record();

  // Do some work
  std::vector<float> data(1000000, 1.0f);
  float *d_data;
  cudaMalloc(&d_data, data.size() * sizeof(float));
  cudaMemcpy(d_data, data.data(), data.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaFree(d_data);

  end.record();
  end.synchronize();

  float elapsed = CudaEvent::elapsedMs(start, end);
  EXPECT_GT(elapsed, 0.0f);
}

// Test: Memory alignment helper
TEST_F(IntegrationTest, MemoryAlignment) {
  EXPECT_EQ(alignTo(100, 128), 128);
  EXPECT_EQ(alignTo(128, 128), 128);
  EXPECT_EQ(alignTo(129, 128), 256);
  EXPECT_EQ(alignTo(0, 128), 0);
}

// Test: Kernel config auto-tuning
TEST_F(IntegrationTest, KernelConfigAutoTune) {
  auto config1 = KernelConfig::autoTune(100);
  EXPECT_GE(config1.block_size, 32);

  auto config2 = KernelConfig::autoTune(10000);
  EXPECT_GE(config2.block_size, 256);
}

// Test: KV Cache with multiple sequences
TEST_F(IntegrationTest, KVCacheMultipleSequences) {
  KVCacheConfig config;
  config.num_layers = 2;
  config.num_heads = 4;
  config.head_dim = 16;
  config.max_seq_len = 32;
  config.max_batch_size = 4;

  KVCacheManager cache(config);

  // Allocate multiple sequences
  std::vector<int> seq_ids;
  for (int i = 0; i < 3; ++i) {
    auto result = cache.allocateSequence(16);
    ASSERT_TRUE(result.isOk());
    seq_ids.push_back(result.value());
  }

  // Verify all sequences are independent
  EXPECT_EQ(seq_ids.size(), 3);
  for (size_t i = 0; i < seq_ids.size(); ++i) {
    for (size_t j = i + 1; j < seq_ids.size(); ++j) {
      EXPECT_NE(seq_ids[i], seq_ids[j]);
    }
  }

  // Release sequences
  for (int seq_id : seq_ids) {
    cache.releaseSequence(seq_id);
  }
}

// Test: Transformer layer forward pass dimensions
TEST_F(IntegrationTest, TransformerLayerDimensions) {
  auto config = createTinyConfig();

  // Create random weights for one layer
  TransformerWeights weights;
  int hidden = config.hidden_dim;
  int kv_dim = config.num_kv_heads * config.head_dim;
  int inter = config.intermediate_dim;
  int group_size = 32;

  weights.wq = createRandomWeight(hidden, hidden, group_size, 100);
  weights.wk = createRandomWeight(hidden, kv_dim, group_size, 101);
  weights.wv = createRandomWeight(hidden, kv_dim, group_size, 102);
  weights.wo = createRandomWeight(hidden, hidden, group_size, 103);
  weights.w1 = createRandomWeight(hidden, inter, group_size, 104);
  weights.w2 = createRandomWeight(inter, hidden, group_size, 105);
  weights.w3 = createRandomWeight(hidden, inter, group_size, 106);
  weights.rms_att_weight = randomDeviceFP16(hidden, 1.0f, 107);
  weights.rms_ffn_weight = randomDeviceFP16(hidden, 1.0f, 108);

  // Create KV cache
  KVCacheConfig kv_config;
  kv_config.num_layers = 1;
  kv_config.num_heads = config.num_kv_heads;
  kv_config.head_dim = config.head_dim;
  kv_config.max_seq_len = config.max_seq_len;
  kv_config.max_batch_size = 1;

  KVCacheManager kv_cache(kv_config);
  auto alloc_result = kv_cache.allocateSequence(32);
  ASSERT_TRUE(alloc_result.isOk());
  int seq_id = alloc_result.value();

  // Create transformer layer
  TransformerLayer layer(0, weights, config);

  // Create input hidden states
  half *hidden_states = randomDeviceFP16(hidden, 1.0f, 200);

  // Forward pass should not crash
  layer.forward(hidden_states, kv_cache, seq_id, 0);
  cudaDeviceSynchronize();

  // Verify KV cache was updated
  EXPECT_EQ(kv_cache.getSeqLen(seq_id), 1);

  // Cleanup
  cudaFree(hidden_states);
  freeWeight(weights.wq);
  freeWeight(weights.wk);
  freeWeight(weights.wv);
  freeWeight(weights.wo);
  freeWeight(weights.w1);
  freeWeight(weights.w2);
  freeWeight(weights.w3);
  cudaFree(weights.rms_att_weight);
  cudaFree(weights.rms_ffn_weight);
}

// Test: Greedy sampling determinism
TEST_F(IntegrationTest, GreedySamplingDeterministic) {
  int vocab_size = 1000;
  std::vector<half> logits(vocab_size);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (auto &v : logits) {
    v = __float2half(dist(gen));
  }

  // Greedy sampling should always return the same result
  int result1 = InferenceEngine::sampleGreedy(logits.data(), vocab_size);
  int result2 = InferenceEngine::sampleGreedy(logits.data(), vocab_size);
  int result3 = InferenceEngine::sampleGreedy(logits.data(), vocab_size);

  EXPECT_EQ(result1, result2);
  EXPECT_EQ(result2, result3);
}

// Test: Temperature affects sampling distribution
TEST_F(IntegrationTest, TemperatureAffectsSampling) {
  int vocab_size = 100;
  std::vector<half> logits(vocab_size);

  // Create logits with a clear peak
  for (int i = 0; i < vocab_size; ++i) {
    logits[i] = __float2half(static_cast<float>(i) / 10.0f);
  }

  // Low temperature should concentrate on high logits
  std::map<int, int> low_temp_counts;
  for (int i = 0; i < 100; ++i) {
    int result =
        InferenceEngine::sampleTemperature(logits.data(), vocab_size, 0.1f, i);
    low_temp_counts[result]++;
  }

  // High temperature should spread more
  std::map<int, int> high_temp_counts;
  for (int i = 0; i < 100; ++i) {
    int result = InferenceEngine::sampleTemperature(logits.data(), vocab_size,
                                                    2.0f, i + 1000);
    high_temp_counts[result]++;
  }

  // Low temperature should have fewer unique values
  EXPECT_LE(low_temp_counts.size(), high_temp_counts.size());
}

// Test: Generation statistics tracking
TEST_F(IntegrationTest, GenerationStatsTracking) {
  GenerationStats stats;

  // Default values
  EXPECT_EQ(stats.prefill_time_ms, 0.0f);
  EXPECT_EQ(stats.decode_time_ms, 0.0f);
  EXPECT_EQ(stats.prompt_tokens, 0);
  EXPECT_EQ(stats.tokens_generated, 0);
  EXPECT_EQ(stats.tokens_per_second, 0.0f);
}
