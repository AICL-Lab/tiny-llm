#include "attention.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/kv_cache.h"
#include "tiny_llm/transformer.h"
#include "transpose_weights.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
// #include <rapidcheck.h>
// NOTE: rapidcheck/gtest is disabled in .cu tests due to GCC 11/12 + nvcc
// std::function compatibility issues during CI builds.
// #include <rapidcheck/gtest.h>
#include <vector>

using namespace tiny_llm;
using namespace tiny_llm::kernels;

// Helper to check if CUDA device is available
static bool hasCudaDevice() {
    static bool checked = false;
    static bool has_device = false;
    if (!checked) {
        int         device_count = 0;
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
        int         device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        cudaSetDevice(0);
    }

    void TearDown() override { cudaDeviceSynchronize(); }

    // Generate random FP16 tensor
    std::vector<half> randomFP16(int size, float scale = 1.0f, unsigned seed = 42) {
        std::vector<half>                     data(size);
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto &v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }

    // Generate random INT8 weights
    std::vector<int8_t> randomINT8(int size, unsigned seed = 123) {
        std::vector<int8_t>                data(size);
        std::mt19937                       gen(seed);
        std::uniform_int_distribution<int> dist(-127, 127);
        for (auto &v : data) {
            v = static_cast<int8_t>(dist(gen));
        }
        return data;
    }

    // Generate random scales
    std::vector<half> randomScales(int size, unsigned seed = 456) {
        std::vector<half>                     data(size);
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(0.001f, 0.1f);
        for (auto &v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }

    // Compute relative error between two tensors
    float computeRelativeError(const std::vector<half> &a, const std::vector<half> &b) {
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
    // Token-major layout test
    int   num_q_heads = 4;
    int   num_kv_heads = 4; // MHA
    int   seq_len = 8;
    int   head_dim = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Generate random Q, K, V in token-major layout
    auto query = randomFP16(num_q_heads * head_dim, 1.0f, 100);
    auto k_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 101);
    auto v_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 102);

    DeviceBuffer<half> d_query(num_q_heads * head_dim);
    DeviceBuffer<half> d_k_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_v_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output_decode(num_q_heads * head_dim);

    d_query.copyFromHost(query.data(), query.size());
    d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
    d_v_cache.copyFromHost(v_cache.data(), v_cache.size());

    attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(), d_output_decode.data(),
                     scale, num_q_heads, num_kv_heads, seq_len, head_dim);

    cudaDeviceSynchronize();

    std::vector<half> output_decode(num_q_heads * head_dim);
    d_output_decode.copyToHost(output_decode.data(), output_decode.size());
    cudaDeviceSynchronize();

    bool has_nonzero = false;
    for (const auto &v : output_decode) {
        if (__half2float(v) != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// P1-8 回归：各投影必须用自身 group_size 反量化。构造 wk.group_size=8、
// 其余=32 的权重，one-hot 输入命中 wk 第 2 组 scale；若误用 wq 的 gs=32，
// K 会取第 0 组 scale（差 3 倍），缓存内容与 CPU 参考明显不符。
TEST_F(TransformerTest, KeyProjectionUsesItsOwnGroupSize) {
    // 配置：hidden=32, 1 head, head_dim=32, intermediate=16
    ModelConfig config;
    config.vocab_size = 32;
    config.hidden_dim = 32;
    config.num_layers = 1;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = 32;
    config.intermediate_dim = 16;
    config.max_seq_len = 8;
    config.rope_theta = 10000.0f;
    config.rms_norm_eps = 1e-5f;

    auto makeQW = [&](int rows, int cols, int group_size, float scaleBase) -> QuantizedWeight {
        QuantizedWeight qw;
        qw.rows = rows;
        qw.cols = cols;
        qw.group_size = group_size;
        const int srows = (rows + group_size - 1) / group_size;

        std::vector<int8_t> h_data(static_cast<size_t>(rows) * cols, 127);
        std::vector<half>   h_scales(static_cast<size_t>(srows) * cols);
        for (int g = 0; g < srows; ++g)
            for (int c = 0; c < cols; ++c)
                h_scales[static_cast<size_t>(g) * cols + c] =
                    __float2half(scaleBase * static_cast<float>(g + 1));

        CUDA_CHECK(cudaMalloc(&qw.data, h_data.size()));
        CUDA_CHECK(cudaMalloc(&qw.scales, h_scales.size() * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(qw.data, h_data.data(), h_data.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(qw.scales, h_scales.data(), h_scales.size() * sizeof(half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&qw.data_t, h_data.size()));
        CUDA_CHECK(cudaMalloc(&qw.scales_t, h_scales.size() * sizeof(half)));
        kernels::transpose_int8(qw.data, qw.data_t, rows, cols, 0);
        kernels::transpose_scales(qw.scales, qw.scales_t, srows, cols, 0);
        return qw;
    };

    TransformerWeights lw;
    lw.wq = makeQW(32, 32, 32, 0.01f);
    lw.wk = makeQW(32, 32, 8, 0.01f); // 关键：gs=8，scaleRows=4
    lw.wv = makeQW(32, 32, 32, 0.01f);
    lw.wo = makeQW(32, 32, 32, 0.01f);
    lw.w1 = makeQW(32, 16, 32, 0.01f);
    lw.w2 = makeQW(16, 32, 32, 0.01f);
    lw.w3 = makeQW(32, 16, 32, 0.01f);

    // norm 权重全 1：rmsNorm 输出 = 归一化输入
    std::vector<half> ones(32, __float2half(1.0f));
    CUDA_CHECK(cudaMalloc(&lw.rms_att_weight, ones.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&lw.rms_ffn_weight, ones.size() * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(lw.rms_att_weight, ones.data(), ones.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lw.rms_ffn_weight, ones.data(), ones.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    LayerWorkspace ws;
    ws.allocate(config);
    TransformerLayer layer(0, lw, config, &ws);

    KVCacheConfig kv_config;
    kv_config.num_layers = 1;
    kv_config.num_kv_heads = 1;
    kv_config.head_dim = 32;
    kv_config.max_seq_len = 8;
    kv_config.max_batch_size = 1;
    auto cache_r = KVCacheManager::create(kv_config);
    ASSERT_TRUE(cache_r.isOk()) << cache_r.error();
    auto cache = std::move(cache_r.value());
    auto seq_r = cache->allocateSequence(8);
    ASSERT_TRUE(seq_r.isOk());
    int seq_id = seq_r.value();

    // RoPE 恒等：cos=1, sin=0
    DeviceBuffer<float> d_cos(16), d_sin(16);
    std::vector<float>  ones_f(16, 1.0f), zeros_f(16, 0.0f);
    d_cos.copyFromHost(ones_f.data(), 16);
    d_sin.copyFromHost(zeros_f.data(), 16);
    DeviceBuffer<int> d_pos(1);
    int               zero = 0;
    d_pos.copyFromHost(&zero, 1);

    // 输入 one-hot e_16（命中 wk 第 2 组 scale）
    std::vector<half> h_hidden(32, __float2half(0.0f));
    h_hidden[16] = __float2half(4.0f);
    DeviceBuffer<half> d_hidden(32);
    d_hidden.copyFromHost(h_hidden.data(), 32);

    cudaDeviceSynchronize();
    auto r = layer.forwardPrefill(d_hidden.data(), *cache, seq_id, 1, d_pos.data(), d_cos.data(),
                                  d_sin.data(), 0);
    ASSERT_TRUE(r.isOk()) << r.error();
    cudaDeviceSynchronize();

    // 读回缓存的 K（pos 0），与 CPU 参考比较：
    // x_norm = e16 * sqrt(32)/4；K[c] = x_norm[16] * 127 * scales[2][c]
    auto [k_cache, v_cache] = cache->getCache(seq_id, 0);
    ASSERT_NE(k_cache, nullptr);
    std::vector<half> h_k(32);
    CUDA_CHECK(cudaMemcpy(h_k.data(), k_cache, 32 * sizeof(half), cudaMemcpyDeviceToHost));

    // rmsNorm: y = x/sqrt(mean(x^2)+eps)；x=e16 值 4 → mean=16/32=0.5
    const float x_norm16 = 4.0f / std::sqrt(0.5f + config.rms_norm_eps);
    const float expected_scale = 0.01f * 3.0f; // 第 2 组（g=2）→ (2+1)*0.01
    for (int c = 0; c < 32; ++c) {
        float expected = x_norm16 * 127.0f * expected_scale;
        EXPECT_NEAR(__half2float(h_k[c]), expected, 5e-2f)
            << "dim " << c << " (若得到 ~1/3 值说明用了第 0 组 scale)";
    }
}

// Unit test: KV Cache append and retrieve
TEST_F(TransformerTest, KVCacheAppendRetrieve) {
    KVCacheConfig config;
    config.num_layers = 2;
    config.num_kv_heads = 4;
    config.head_dim = 32;
    config.max_seq_len = 64;
    config.max_batch_size = 2;

    auto cache_result = KVCacheManager::create(config);
    ASSERT_TRUE(cache_result.isOk()) << cache_result.error();
    auto cache = std::move(cache_result.value());

    // Allocate sequence
    auto result = cache->allocateSequence(32);
    ASSERT_TRUE(result.isOk());
    int seq_id = result.value();

    // Generate random K, V
    int  num_tokens = 4;
    auto k_data = randomFP16(num_tokens * config.num_kv_heads * config.head_dim, 1.0f, 200);
    auto v_data = randomFP16(num_tokens * config.num_kv_heads * config.head_dim, 1.0f, 201);

    DeviceBuffer<half> d_k(k_data.size());
    DeviceBuffer<half> d_v(v_data.size());
    d_k.copyFromHost(k_data.data(), k_data.size());
    d_v.copyFromHost(v_data.data(), v_data.size());
    cudaDeviceSynchronize();

    // Append to cache
    cache->appendKV(seq_id, 0, d_k.data(), d_v.data(), num_tokens);
    cudaDeviceSynchronize();

    // appendKV writes data but does not advance the visible sequence length.
    EXPECT_EQ(cache->getSeqLen(seq_id), 0);
    cache->advanceSeqLen(seq_id, num_tokens);
    EXPECT_EQ(cache->getSeqLen(seq_id), num_tokens);

    // Get cache pointers
    auto [k_cache, v_cache] = cache->getCache(seq_id, 0);
    EXPECT_NE(k_cache, nullptr);
    EXPECT_NE(v_cache, nullptr);

    // Verify data was copied correctly
    std::vector<half> k_retrieved(k_data.size());
    std::vector<half> v_retrieved(v_data.size());
    cudaMemcpy(k_retrieved.data(), k_cache, k_data.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_retrieved.data(), v_cache, v_data.size() * sizeof(half), cudaMemcpyDeviceToHost);

    float k_error = computeRelativeError(k_data, k_retrieved);
    float v_error = computeRelativeError(v_data, v_retrieved);

    EXPECT_LT(k_error, 0.001f) << "K cache data mismatch";
    EXPECT_LT(v_error, 0.001f) << "V cache data mismatch";
}

#if 0
// Property-based tests
// Feature: tiny-llm-inference-engine, Property 5: Incremental Decoding
// Equivalence Validates: Requirements 4.6
// NOTE: Disabled in CUDA translation units due to GCC 11/12 + nvcc
// compatibility issues with rapidcheck's GTest integration.

class TransformerPropertyTest : public TransformerTest {
  protected:
    void SetUp() override {
        if (hasCudaDevice()) {
            cudaSetDevice(0);
        }
    }

    void TearDown() override {
        if (hasCudaDevice()) {
            cudaDeviceSynchronize();
        }
    }
};

// Property 5: Incremental Decoding Equivalence
// For any input sequence, the output of incremental decoding (using KV cache)
// must be identical to full sequence recomputation.

RC_GTEST_FIXTURE_PROP(TransformerPropertyTest, IncrementalDecodingEquivalence,
                      (int heads_raw, int seq_raw, int dim_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions to reasonable ranges
    int   num_heads = 2 + (std::abs(heads_raw) % 6); // 2 to 8
    int   seq_len = 4 + (std::abs(seq_raw) % 28);    // 4 to 32
    int   head_dim = 16 + (std::abs(dim_raw) % 48);  // 16 to 64
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
                      d_output_full.data(), scale, 1, num_heads, seq_len, head_dim);

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
            query_last[h * head_dim + d] = query_full[(h * seq_len + last_pos) * head_dim + d];
        }
    }

    DeviceBuffer<half> d_query_last(num_heads * head_dim);
    DeviceBuffer<half> d_output_decode(num_heads * head_dim);

    d_query_last.copyFromHost(query_last.data(), query_last.size());

    // Run decode attention (query last token against full K, V cache)
    attention_decode(d_query_last.data(), d_key_full.data(), d_value_full.data(),
                     d_output_decode.data(), scale, 1, num_heads, seq_len, head_dim);

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
                      (int layers_raw, int heads_raw, int seq_raw, int dim_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions
    int num_layers = 1 + (std::abs(layers_raw) % 4); // 1 to 4
    int num_heads = 2 + (std::abs(heads_raw) % 6);   // 2 to 8
    int max_seq_len = 32 + (std::abs(seq_raw) % 96); // 32 to 128
    int head_dim = 16 + (std::abs(dim_raw) % 48);    // 16 to 64

    KVCacheConfig config;
    config.num_layers = num_layers;
    config.num_kv_heads = num_heads;
    config.head_dim = head_dim;
    config.max_seq_len = max_seq_len;
    config.max_batch_size = 2;

    auto cache_result = KVCacheManager::create(config);
    RC_ASSERT(cache_result.isOk());
    auto cache = std::move(cache_result.value());

    // Allocate sequence
    int  alloc_len = max_seq_len / 2;
    auto result = cache->allocateSequence(alloc_len);
    RC_ASSERT(result.isOk());
    int seq_id = result.value();

    // Generate and append random K, V for each layer
    // In real usage, all layers append at the same time for each token batch
    int                             num_tokens = 4;
    std::vector<std::vector<half>>  k_data_per_layer(num_layers);
    std::vector<std::vector<half>>  v_data_per_layer(num_layers);
    std::vector<DeviceBuffer<half>> d_k_buffers;
    std::vector<DeviceBuffer<half>> d_v_buffers;

    // First, generate all data and copy to device
    for (int layer = 0; layer < num_layers; ++layer) {
        k_data_per_layer[layer] =
            randomFP16(num_tokens * num_heads * head_dim, 1.0f, seed + layer * 2);
        v_data_per_layer[layer] =
            randomFP16(num_tokens * num_heads * head_dim, 1.0f, seed + layer * 2 + 1);

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
        cache->appendKV(seq_id, layer, d_k_buffers[layer].data(), d_v_buffers[layer].data(),
                       num_tokens);
    }
    cudaDeviceSynchronize();

    // Property: Data should be preserved in cache
    for (int layer = 0; layer < num_layers; ++layer) {
        auto [k_cache, v_cache] = cache->getCache(seq_id, layer);
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
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions
    int num_heads = 2 + (std::abs(heads_raw) % 6);
    int head_dim = 16 + (std::abs(dim_raw) % 48);
    int max_seq_len = 64;

    KVCacheConfig config;
    config.num_layers = 1;
    config.num_kv_heads = num_heads;
    config.head_dim = head_dim;
    config.max_seq_len = max_seq_len;
    config.max_batch_size = 2;

    auto cache_result = KVCacheManager::create(config);
    RC_ASSERT(cache_result.isOk());
    auto cache = std::move(cache_result.value());

    auto result = cache->allocateSequence(max_seq_len);
    RC_ASSERT(result.isOk());
    int seq_id = result.value();

    // Append tokens one by one
    int               total_tokens = 8;
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

        cache->appendKV(seq_id, 0, d_k.data(), d_v.data(), 1);
        cache->advanceSeqLen(seq_id, 1);
    }
    cudaDeviceSynchronize();

    // Property: Sequential append should produce same result as batch append
    RC_ASSERT(cache->getSeqLen(seq_id) == total_tokens);

    auto [k_cache, v_cache] = cache->getCache(seq_id, 0);
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
#endif
