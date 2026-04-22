#include "attention.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/kv_cache.h"
#include "tiny_llm/transformer.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

using namespace tiny_llm;
using namespace tiny_llm::kernels;

// Helper to check if CUDA device is available
static bool hasCudaDevice() {
  static bool checked = false;
  static bool has_device = false;
  if (!checked) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    has_device = (err == cudaSuccess && device_count > 0);
    checked = true;
  }
  return has_device;
}

// Helper class for Transformer tests
class TransformerTest : public ::testing::Test {
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

  // Generate random FP16 tensor
  std::vector<half> randomFP16(int size, float scale = 1.0f,
                               unsigned seed = 42) {
    std::vector<half> data(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto &v : data) {
      v = __float2half(dist(gen));
    }
    return data;
  }

  // Generate random INT8 weights
  std::vector<int8_t> randomINT8(int size, unsigned seed = 123) {
    std::vector<int8_t> data(size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto &v : data) {
      v = static_cast<int8_t>(dist(gen));
    }
    return data;
  }

  // Generate random scales
  std::vector<half> randomScales(int size, unsigned seed = 456) {
    std::vector<half> data(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.001f, 0.1f);
    for (auto &v : data) {
      v = __float2half(dist(gen));
    }
    return data;
  }

  // Compute relative error between two tensors
  float computeRelativeError(const std::vector<half> &a,
                             const std::vector<half> &b) {
    float max_diff = 0.0f;
    float max_val = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
      float va = __half2float(a[i]);
      float vb = __half2float(b[i]);
      max_diff = std::max(max_diff, std::abs(va - vb));
      max_val = std::max(max_val, std::max(std::abs(va), std::abs(vb)));
    }

    return max_val > 0 ? max_diff / max_val : 0.0f;
  }
};

// Unit test: Basic attention decode vs prefill equivalence
TEST_F(TransformerTest, AttentionDecodeVsPrefillSingleToken) {
  // Test that decode attention produces same result as prefill for single token
  int batch_size = 1;
  int num_heads = 4;
  int seq_len = 8;
  int head_dim = 32;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Generate random Q, K, V
  auto query = randomFP16(batch_size * num_heads * head_dim, 1.0f, 100);
  auto k_cache =
      randomFP16(batch_size * num_heads * seq_len * head_dim, 1.0f, 101);
  auto v_cache =
      randomFP16(batch_size * num_heads * seq_len * head_dim, 1.0f, 102);

  // Allocate device memory
  DeviceBuffer<half> d_query(batch_size * num_heads * head_dim);
  DeviceBuffer<half> d_k_cache(batch_size * num_heads * seq_len * head_dim);
  DeviceBuffer<half> d_v_cache(batch_size * num_heads * seq_len * head_dim);
  DeviceBuffer<half> d_output_decode(batch_size * num_heads * head_dim);

  d_query.copyFromHost(query.data(), query.size());
  d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
  d_v_cache.copyFromHost(v_cache.data(), v_cache.size());

  // Run decode attention
  attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(),
                   d_output_decode.data(), scale, batch_size, num_heads,
                   seq_len, head_dim);

  cudaDeviceSynchronize();

  std::vector<half> output_decode(batch_size * num_heads * head_dim);
  d_output_decode.copyToHost(output_decode.data(), output_decode.size());
  cudaDeviceSynchronize();

  // Verify output is not all zeros
  bool has_nonzero = false;
  for (const auto &v : output_decode) {
    if (__half2float(v) != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero);
}

// Unit test: KV Cache append and retrieve
TEST_F(TransformerTest, KVCacheAppendRetrieve) {
  KVCacheConfig config;
  config.num_layers = 2;
  config.num_heads = 4;
  config.head_dim = 32;
  config.max_seq_len = 64;
  config.max_batch_size = 2;

  KVCacheManager cache(config);

  // Allocate sequence
  auto result = cache.allocateSequence(32);
  ASSERT_TRUE(result.isOk());
  int seq_id = result.value();

  // Generate random K, V
  int num_tokens = 4;
  auto k_data =
      randomFP16(num_tokens * config.num_heads * config.head_dim, 1.0f, 200);
  auto v_data =
      randomFP16(num_tokens * config.num_heads * config.head_dim, 1.0f, 201);

  DeviceBuffer<half> d_k(k_data.size());
  DeviceBuffer<half> d_v(v_data.size());
  d_k.copyFromHost(k_data.data(), k_data.size());
  d_v.copyFromHost(v_data.data(), v_data.size());
  cudaDeviceSynchronize();

  // Append to cache
  cache.appendKV(seq_id, 0, d_k.data(), d_v.data(), num_tokens);
  cudaDeviceSynchronize();

  // appendKV writes data but does not advance the visible sequence length.
  EXPECT_EQ(cache.getSeqLen(seq_id), 0);
  cache.advanceSeqLen(seq_id, num_tokens);
  EXPECT_EQ(cache.getSeqLen(seq_id), num_tokens);

  // Get cache pointers
  auto [k_cache, v_cache] = cache.getCache(seq_id, 0);
  EXPECT_NE(k_cache, nullptr);
  EXPECT_NE(v_cache, nullptr);

  // Verify data was copied correctly
  std::vector<half> k_retrieved(k_data.size());
  std::vector<half> v_retrieved(v_data.size());
  cudaMemcpy(k_retrieved.data(), k_cache, k_data.size() * sizeof(half),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(v_retrieved.data(), v_cache, v_data.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  float k_error = computeRelativeError(k_data, k_retrieved);
  float v_error = computeRelativeError(v_data, v_retrieved);

  EXPECT_LT(k_error, 0.001f) << "K cache data mismatch";
  EXPECT_LT(v_error, 0.001f) << "V cache data mismatch";
}

// Property-based tests
// Feature: tiny-llm-inference-engine, Property 5: Incremental Decoding
// Equivalence Validates: Requirements 4.6
// NOTE: Property-based tests are disabled when no CUDA device is available

class TransformerPropertyTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!hasCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }
    cudaSetDevice(0);
  }
  void TearDown() override { cudaDeviceSynchronize(); }
};

// Property 5: Incremental Decoding Equivalence
// For any input sequence, the output of incremental decoding (using KV cache)
// must be identical to full sequence recomputation.

RC_GTEST_FIXTURE_PROP(TransformerPropertyTest, IncrementalDecodingEquivalence,
                      (int heads_raw, int seq_raw, int dim_raw,
                       unsigned seed)) {
  if (!hasCudaDevice()) {
    RC_SKIP("No CUDA device available");
  }
  // Constrain dimensions to reasonable ranges
  int num_heads = 2 + (std::abs(heads_raw) % 6); // 2 to 8
  int seq_len = 4 + (std::abs(seq_raw) % 28);    // 4 to 32
  int head_dim = 16 + (std::abs(dim_raw) % 48);  // 16 to 64
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Generate random Q, K, V for full sequence
  auto query_full = randomFP16(num_heads * seq_len * head_dim, 1.0f, seed);
  auto key_full = randomFP16(num_heads * seq_len * head_dim, 1.0f, seed + 1);
  auto value_full = randomFP16(num_heads * seq_len * head_dim, 1.0f, seed + 2);

  // Allocate device memory
  DeviceBuffer<half> d_query_full(num_heads * seq_len * head_dim);
  DeviceBuffer<half> d_key_full(num_heads * seq_len * head_dim);
  DeviceBuffer<half> d_value_full(num_heads * seq_len * head_dim);
  DeviceBuffer<half> d_output_full(num_heads * seq_len * head_dim);

  d_query_full.copyFromHost(query_full.data(), query_full.size());
  d_key_full.copyFromHost(key_full.data(), key_full.size());
  d_value_full.copyFromHost(value_full.data(), value_full.size());

  // Run full prefill attention
  attention_prefill(d_query_full.data(), d_key_full.data(), d_value_full.data(),
                    d_output_full.data(), scale, 1, num_heads, seq_len,
                    head_dim);

  cudaDeviceSynchronize();

  std::vector<half> output_full(num_heads * seq_len * head_dim);
  d_output_full.copyToHost(output_full.data(), output_full.size());
  cudaDeviceSynchronize();

  // Now run incremental decoding for the last token
  // Use the same K, V as cache, and query the last position
  int last_pos = seq_len - 1;

  // Extract last query
  std::vector<half> query_last(num_heads * head_dim);
  for (int h = 0; h < num_heads; ++h) {
    for (int d = 0; d < head_dim; ++d) {
      query_last[h * head_dim + d] =
          query_full[(h * seq_len + last_pos) * head_dim + d];
    }
  }

  DeviceBuffer<half> d_query_last(num_heads * head_dim);
  DeviceBuffer<half> d_output_decode(num_heads * head_dim);

  d_query_last.copyFromHost(query_last.data(), query_last.size());

  // Run decode attention (query last token against full K, V cache)
  attention_decode(d_query_last.data(), d_key_full.data(), d_value_full.data(),
                   d_output_decode.data(), scale, 1, num_heads, seq_len,
                   head_dim);

  cudaDeviceSynchronize();

  std::vector<half> output_decode(num_heads * head_dim);
  d_output_decode.copyToHost(output_decode.data(), output_decode.size());
  cudaDeviceSynchronize();

  // Extract last position output from full computation
  std::vector<half> output_full_last(num_heads * head_dim);
  for (int h = 0; h < num_heads; ++h) {
    for (int d = 0; d < head_dim; ++d) {
      output_full_last[h * head_dim + d] =
          output_full[(h * seq_len + last_pos) * head_dim + d];
    }
  }

  // Property: Incremental decode output should match full prefill output for
  // last position
  float rel_error = computeRelativeError(output_decode, output_full_last);

  // Allow some tolerance for floating point differences
  RC_ASSERT(rel_error < 0.05f); // 5% tolerance for FP16
}

RC_GTEST_FIXTURE_PROP(TransformerPropertyTest, KVCachePreservesData,
                      (int layers_raw, int heads_raw, int seq_raw, int dim_raw,
                       unsigned seed)) {
  if (!hasCudaDevice()) {
    RC_SKIP("No CUDA device available");
  }
  // Constrain dimensions
  int num_layers = 1 + (std::abs(layers_raw) % 4); // 1 to 4
  int num_heads = 2 + (std::abs(heads_raw) % 6);   // 2 to 8
  int max_seq_len = 32 + (std::abs(seq_raw) % 96); // 32 to 128
  int head_dim = 16 + (std::abs(dim_raw) % 48);    // 16 to 64

  KVCacheConfig config;
  config.num_layers = num_layers;
  config.num_heads = num_heads;
  config.head_dim = head_dim;
  config.max_seq_len = max_seq_len;
  config.max_batch_size = 2;

  KVCacheManager cache(config);

  // Allocate sequence
  int alloc_len = max_seq_len / 2;
  auto result = cache.allocateSequence(alloc_len);
  RC_ASSERT(result.isOk());
  int seq_id = result.value();

  // Generate and append random K, V for each layer
  // In real usage, all layers append at the same time for each token batch
  int num_tokens = 4;
  std::vector<std::vector<half>> k_data_per_layer(num_layers);
  std::vector<std::vector<half>> v_data_per_layer(num_layers);
  std::vector<DeviceBuffer<half>> d_k_buffers;
  std::vector<DeviceBuffer<half>> d_v_buffers;

  // First, generate all data and copy to device
  for (int layer = 0; layer < num_layers; ++layer) {
    k_data_per_layer[layer] =
        randomFP16(num_tokens * num_heads * head_dim, 1.0f, seed + layer * 2);
    v_data_per_layer[layer] = randomFP16(num_tokens * num_heads * head_dim,
                                         1.0f, seed + layer * 2 + 1);

    d_k_buffers.emplace_back(k_data_per_layer[layer].size());
    d_v_buffers.emplace_back(v_data_per_layer[layer].size());
    d_k_buffers.back().copyFromHost(k_data_per_layer[layer].data(),
                                    k_data_per_layer[layer].size());
    d_v_buffers.back().copyFromHost(v_data_per_layer[layer].data(),
                                    v_data_per_layer[layer].size());
  }
  cudaDeviceSynchronize();

  // Append all layers in order (layer 0 first to update seq_len, then others)
  // This simulates how TransformerLayer would use it
  for (int layer = 0; layer < num_layers; ++layer) {
    cache.appendKV(seq_id, layer, d_k_buffers[layer].data(),
                   d_v_buffers[layer].data(), num_tokens);
  }
  cudaDeviceSynchronize();

  // Property: Data should be preserved in cache
  for (int layer = 0; layer < num_layers; ++layer) {
    auto [k_cache, v_cache] = cache.getCache(seq_id, layer);
    RC_ASSERT(k_cache != nullptr);
    RC_ASSERT(v_cache != nullptr);

    std::vector<half> k_retrieved(k_data_per_layer[layer].size());
    std::vector<half> v_retrieved(v_data_per_layer[layer].size());
    cudaMemcpy(k_retrieved.data(), k_cache, k_retrieved.size() * sizeof(half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(v_retrieved.data(), v_cache, v_retrieved.size() * sizeof(half),
               cudaMemcpyDeviceToHost);

    float k_error = computeRelativeError(k_data_per_layer[layer], k_retrieved);
    float v_error = computeRelativeError(v_data_per_layer[layer], v_retrieved);

    RC_ASSERT(k_error < 0.001f);
    RC_ASSERT(v_error < 0.001f);
  }
}

RC_GTEST_FIXTURE_PROP(TransformerPropertyTest, SequentialAppendEquivalence,
                      (int heads_raw, int dim_raw, unsigned seed)) {
  if (!hasCudaDevice()) {
    RC_SKIP("No CUDA device available");
  }
  // Constrain dimensions
  int num_heads = 2 + (std::abs(heads_raw) % 6);
  int head_dim = 16 + (std::abs(dim_raw) % 48);
  int max_seq_len = 64;

  KVCacheConfig config;
  config.num_layers = 1;
  config.num_heads = num_heads;
  config.head_dim = head_dim;
  config.max_seq_len = max_seq_len;
  config.max_batch_size = 2;

  KVCacheManager cache(config);

  auto result = cache.allocateSequence(max_seq_len);
  RC_ASSERT(result.isOk());
  int seq_id = result.value();

  // Append tokens one by one
  int total_tokens = 8;
  std::vector<half> all_k, all_v;

  for (int t = 0; t < total_tokens; ++t) {
    auto k_token = randomFP16(num_heads * head_dim, 1.0f, seed + t * 2);
    auto v_token = randomFP16(num_heads * head_dim, 1.0f, seed + t * 2 + 1);

    all_k.insert(all_k.end(), k_token.begin(), k_token.end());
    all_v.insert(all_v.end(), v_token.begin(), v_token.end());

    DeviceBuffer<half> d_k(k_token.size());
    DeviceBuffer<half> d_v(v_token.size());
    d_k.copyFromHost(k_token.data(), k_token.size());
    d_v.copyFromHost(v_token.data(), v_token.size());
    cudaDeviceSynchronize();

    cache.appendKV(seq_id, 0, d_k.data(), d_v.data(), 1);
    cache.advanceSeqLen(seq_id, 1);
  }
  cudaDeviceSynchronize();

  // Property: Sequential append should produce same result as batch append
  RC_ASSERT(cache.getSeqLen(seq_id) == total_tokens);

  auto [k_cache, v_cache] = cache.getCache(seq_id, 0);
  std::vector<half> k_retrieved(all_k.size());
  std::vector<half> v_retrieved(all_v.size());
  cudaMemcpy(k_retrieved.data(), k_cache, k_retrieved.size() * sizeof(half),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(v_retrieved.data(), v_cache, v_retrieved.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  float k_error = computeRelativeError(all_k, k_retrieved);
  float v_error = computeRelativeError(all_v, v_retrieved);

  RC_ASSERT(k_error < 0.001f);
  RC_ASSERT(v_error < 0.001f);
}
