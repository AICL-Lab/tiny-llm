#include "rmsnorm.cuh"
#include "rope.cuh"
#include "paged_kv.cuh"
#include "elementwise.cuh"
#include "tiny_llm/cuda_utils.h"
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

// Helper class for GPU test fixtures
class RMSNormTest : public ::testing::Test {
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

    // Generate random FP16 vector
    std::vector<half> randomFP16(int size, float scale = 1.0f, unsigned seed = 42) {
        std::vector<half>                     data(size);
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto &v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }

    // Generate FP16 weights (positive values for RMSNorm weights)
    std::vector<half> randomWeights(int size, unsigned seed = 123) {
        std::vector<half>                     data(size);
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(0.5f, 1.5f);
        for (auto &v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }

    // Compute RMS of a vector (on host)
    float computeRMS(const std::vector<half> &data) {
        float sum_sq = 0.0f;
        for (const auto &v : data) {
            float val = __half2float(v);
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq / data.size());
    }

    // Compute RMS for a specific row in a batch
    float computeRowRMS(const std::vector<half> &data, int row, int hidden_dim) {
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            float val = __half2float(data[row * hidden_dim + i]);
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq / hidden_dim);
    }
};

// Unit tests
TEST_F(RMSNormTest, BasicCorrectness) {
    int   batch_size = 2;
    int   hidden_dim = 64;
    float eps = 1e-5f;

    auto input = randomFP16(batch_size * hidden_dim, 1.0f);
    auto weight = randomWeights(hidden_dim);

    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);

    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);

    rmsnorm(d_input.data(), d_weight.data(), d_output.data(), batch_size, hidden_dim, eps);

    cudaDeviceSynchronize();

    std::vector<half> output(batch_size * hidden_dim);
    d_output.copyToHost(output.data(), batch_size * hidden_dim);
    cudaDeviceSynchronize();

    // Verify output is not all zeros
    bool has_nonzero = false;
    for (const auto &v : output) {
        if (__half2float(v) != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(RMSNormTest, SmallHiddenDim) {
    int   batch_size = 4;
    int   hidden_dim = 32;
    float eps = 1e-5f;

    auto input = randomFP16(batch_size * hidden_dim, 1.0f);
    auto weight = randomWeights(hidden_dim);

    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);

    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);

    EXPECT_NO_THROW({
        rmsnorm(d_input.data(), d_weight.data(), d_output.data(), batch_size, hidden_dim, eps);
        cudaDeviceSynchronize();
    });
}

TEST_F(RMSNormTest, LargeHiddenDim) {
    int   batch_size = 2;
    int   hidden_dim = 4096;
    float eps = 1e-5f;

    auto input = randomFP16(batch_size * hidden_dim, 1.0f);
    auto weight = randomWeights(hidden_dim);

    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);

    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);

    EXPECT_NO_THROW({
        rmsnorm(d_input.data(), d_weight.data(), d_output.data(), batch_size, hidden_dim, eps);
        cudaDeviceSynchronize();
    });
}

TEST_F(RMSNormTest, InPlaceVersion) {
    int   batch_size = 2;
    int   hidden_dim = 128;
    float eps = 1e-5f;

    auto input = randomFP16(batch_size * hidden_dim, 1.0f);
    auto weight = randomWeights(hidden_dim);

    DeviceBuffer<half> d_x(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);

    d_x.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);

    EXPECT_NO_THROW({
        rmsnorm_inplace(d_x.data(), d_weight.data(), batch_size, hidden_dim, eps);
        cudaDeviceSynchronize();
    });
}

#if 0
// Property-based tests
// Feature: tiny-llm-inference-engine, Property 4: RMSNorm Output Properties
// Validates: Requirements 4.4
// NOTE: Disabled in CUDA translation units due to GCC 11/12 + nvcc
// compatibility issues with rapidcheck's GTest integration.

class RMSNormPropertyTest : public RMSNormTest {
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

// Property 4: RMSNorm Output Properties
// For any input tensor x, the RMSNorm output y (before weight multiplication)
// should satisfy: sqrt(mean(y^2)) ≈ 1.0 (within floating point tolerance)
//
// Since the kernel applies weight multiplication, we test with unit weights to
// verify the normalization property, then test with random weights to verify
// weight application.

RC_GTEST_FIXTURE_PROP(RMSNormPropertyTest, OutputRMSIsOne,
                      (int batch_raw, int dim_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions to reasonable ranges
    int   batch_size = 1 + (std::abs(batch_raw) % 16);
    int   hidden_dim = 32 + (std::abs(dim_raw) % 2016); // 32 to 2048
    float eps = 1e-5f;

    // Generate random input
    auto input = randomFP16(batch_size * hidden_dim, 2.0f, seed);

    // Use unit weights to test pure normalization
    std::vector<half> unit_weight(hidden_dim);
    for (auto &w : unit_weight) {
        w = __float2half(1.0f);
    }

    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);

    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(unit_weight.data(), hidden_dim);

    rmsnorm(d_input.data(), d_weight.data(), d_output.data(), batch_size, hidden_dim, eps);

    cudaDeviceSynchronize();

    std::vector<half> output(batch_size * hidden_dim);
    d_output.copyToHost(output.data(), batch_size * hidden_dim);
    cudaDeviceSynchronize();

    // Property: For each row, RMS of output should be approximately 1.0
    for (int b = 0; b < batch_size; ++b) {
        float rms = computeRowRMS(output, b, hidden_dim);

        // Allow tolerance for FP16 precision
        // RMS should be close to 1.0 (within 5% for FP16)
        RC_ASSERT(rms > 0.95f && rms < 1.05f);
    }
}

RC_GTEST_FIXTURE_PROP(RMSNormPropertyTest, WeightScaling,
                      (int batch_raw, int dim_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions
    int   batch_size = 1 + (std::abs(batch_raw) % 8);
    int   hidden_dim = 64 + (std::abs(dim_raw) % 448); // 64 to 512
    float eps = 1e-5f;

    // Generate random input
    auto input = randomFP16(batch_size * hidden_dim, 1.0f, seed);

    // Test with constant weight = 2.0
    std::vector<half> const_weight(hidden_dim);
    float             weight_val = 2.0f;
    for (auto &w : const_weight) {
        w = __float2half(weight_val);
    }

    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);

    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(const_weight.data(), hidden_dim);

    rmsnorm(d_input.data(), d_weight.data(), d_output.data(), batch_size, hidden_dim, eps);

    cudaDeviceSynchronize();

    std::vector<half> output(batch_size * hidden_dim);
    d_output.copyToHost(output.data(), batch_size * hidden_dim);
    cudaDeviceSynchronize();

    // Property: With constant weight w, RMS of output should be approximately w
    for (int b = 0; b < batch_size; ++b) {
        float rms = computeRowRMS(output, b, hidden_dim);

        // RMS should be close to weight_val (within 10% for FP16)
        RC_ASSERT(rms > weight_val * 0.9f && rms < weight_val * 1.1f);
    }
}

RC_GTEST_FIXTURE_PROP(RMSNormPropertyTest, NonZeroOutput,
                      (int batch_raw, int dim_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions
    int   batch_size = 1 + (std::abs(batch_raw) % 8);
    int   hidden_dim = 32 + (std::abs(dim_raw) % 480);
    float eps = 1e-5f;

    // Generate non-zero input
    auto input = randomFP16(batch_size * hidden_dim, 1.0f, seed);
    auto weight = randomWeights(hidden_dim, seed + 1);

    // Ensure input is not all zeros
    bool input_has_nonzero = false;
    for (const auto &v : input) {
        if (std::abs(__half2float(v)) > 1e-6f) {
            input_has_nonzero = true;
            break;
        }
    }
    RC_PRE(input_has_nonzero);

    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);

    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);

    rmsnorm(d_input.data(), d_weight.data(), d_output.data(), batch_size, hidden_dim, eps);

    cudaDeviceSynchronize();

    std::vector<half> output(batch_size * hidden_dim);
    d_output.copyToHost(output.data(), batch_size * hidden_dim);
    cudaDeviceSynchronize();

    // Property: Output should have non-zero values
    bool has_nonzero = false;
    for (const auto &v : output) {
        if (std::abs(__half2float(v)) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    RC_ASSERT(has_nonzero);
}

RC_GTEST_FIXTURE_PROP(RMSNormPropertyTest, InPlaceEquivalence,
                      (int batch_raw, int dim_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions
    int   batch_size = 1 + (std::abs(batch_raw) % 8);
    int   hidden_dim = 64 + (std::abs(dim_raw) % 448);
    float eps = 1e-5f;

    // Generate random input
    auto input = randomFP16(batch_size * hidden_dim, 1.0f, seed);
    auto weight = randomWeights(hidden_dim, seed + 1);

    // Run out-of-place version
    DeviceBuffer<half> d_input1(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);

    d_input1.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);

    rmsnorm(d_input1.data(), d_weight.data(), d_output.data(), batch_size, hidden_dim, eps);

    // Run in-place version
    DeviceBuffer<half> d_input2(batch_size * hidden_dim);
    d_input2.copyFromHost(input.data(), batch_size * hidden_dim);

    rmsnorm_inplace(d_input2.data(), d_weight.data(), batch_size, hidden_dim, eps);

    cudaDeviceSynchronize();

    std::vector<half> output_oop(batch_size * hidden_dim);
    std::vector<half> output_ip(batch_size * hidden_dim);
    d_output.copyToHost(output_oop.data(), batch_size * hidden_dim);
    d_input2.copyToHost(output_ip.data(), batch_size * hidden_dim);
    cudaDeviceSynchronize();

    // Property: In-place and out-of-place should produce same results
    for (int i = 0; i < batch_size * hidden_dim; ++i) {
        float v1 = __half2float(output_oop[i]);
        float v2 = __half2float(output_ip[i]);
        float diff = std::abs(v1 - v2);
        float max_val = std::max(std::abs(v1), std::abs(v2));
        float rel_diff = max_val > 1e-6f ? diff / max_val : diff;

        // Allow small tolerance for floating point differences
        RC_ASSERT(rel_diff < 0.01f);
    }
}
#endif

// ============================================================================
// Attention Kernel Tests
// ============================================================================

#include "attention.cuh"

class AttentionTest : public ::testing::Test {
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
};

// Unit tests for Attention
TEST_F(AttentionTest, BasicDecodeAttention) {
    // Token-major layout: Q [1,Hq,D], K_cache [S,Hkv,D], V_cache [S,Hkv,D]
    int   num_q_heads = 2;
    int   num_kv_heads = 2; // MHA: Hq == Hkv
    int   seq_len = 8;
    int   head_dim = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto query = randomFP16(num_q_heads * head_dim);
    auto k_cache = randomFP16(seq_len * num_kv_heads * head_dim);
    auto v_cache = randomFP16(seq_len * num_kv_heads * head_dim);

    DeviceBuffer<half> d_query(num_q_heads * head_dim);
    DeviceBuffer<half> d_k_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_v_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(num_q_heads * head_dim);

    d_query.copyFromHost(query.data(), query.size());
    d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
    d_v_cache.copyFromHost(v_cache.data(), v_cache.size());

    EXPECT_NO_THROW({
        attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(), d_output.data(), scale,
                         num_q_heads, num_kv_heads, seq_len, head_dim);
        cudaDeviceSynchronize();
    });

    std::vector<half> output(num_q_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    bool has_nonzero = false;
    for (const auto &v : output) {
        if (__half2float(v) != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// GQA decode: Hq=4, Hkv=2, group_size=2
TEST_F(AttentionTest, GQADecodeAttention) {
    int   num_q_heads = 4;
    int   num_kv_heads = 2;
    int   seq_len = 8;
    int   head_dim = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto query = randomFP16(num_q_heads * head_dim, 1.0f, 200);
    auto k_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 201);
    auto v_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 202);

    DeviceBuffer<half> d_query(num_q_heads * head_dim);
    DeviceBuffer<half> d_k_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_v_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(num_q_heads * head_dim);

    d_query.copyFromHost(query.data(), query.size());
    d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
    d_v_cache.copyFromHost(v_cache.data(), v_cache.size());

    EXPECT_NO_THROW({
        attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(), d_output.data(), scale,
                         num_q_heads, num_kv_heads, seq_len, head_dim);
        cudaDeviceSynchronize();
    });

    std::vector<half> output(num_q_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    bool has_nonzero = false;
    for (const auto &v : output) {
        if (__half2float(v) != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(AttentionTest, BasicPrefillAttention) {
    // Token-major layout: Q [S,Hq,D], K [S,Hkv,D], V [S,Hkv,D]
    int   num_q_heads = 2;
    int   num_kv_heads = 2;
    int   seq_len = 8;
    int   head_dim = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto query = randomFP16(seq_len * num_q_heads * head_dim);
    auto key = randomFP16(seq_len * num_kv_heads * head_dim);
    auto value = randomFP16(seq_len * num_kv_heads * head_dim);

    DeviceBuffer<half> d_query(seq_len * num_q_heads * head_dim);
    DeviceBuffer<half> d_key(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_value(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(seq_len * num_q_heads * head_dim);

    d_query.copyFromHost(query.data(), query.size());
    d_key.copyFromHost(key.data(), key.size());
    d_value.copyFromHost(value.data(), value.size());

    EXPECT_NO_THROW({
        attention_prefill(d_query.data(), d_key.data(), d_value.data(), d_output.data(), scale,
                          num_q_heads, num_kv_heads, seq_len, head_dim);
        cudaDeviceSynchronize();
    });
}

// GQA prefill: Hq=4, Hkv=2
TEST_F(AttentionTest, GQAPrefillAttention) {
    int   num_q_heads = 4;
    int   num_kv_heads = 2;
    int   seq_len = 8;
    int   head_dim = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto query = randomFP16(seq_len * num_q_heads * head_dim, 1.0f, 300);
    auto key = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 301);
    auto value = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 302);

    DeviceBuffer<half> d_query(seq_len * num_q_heads * head_dim);
    DeviceBuffer<half> d_key(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_value(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(seq_len * num_q_heads * head_dim);

    d_query.copyFromHost(query.data(), query.size());
    d_key.copyFromHost(key.data(), key.size());
    d_value.copyFromHost(value.data(), value.size());

    EXPECT_NO_THROW({
        attention_prefill(d_query.data(), d_key.data(), d_value.data(), d_output.data(), scale,
                          num_q_heads, num_kv_heads, seq_len, head_dim);
        cudaDeviceSynchronize();
    });
}

// ============================================================================
// 任务 A1：QKV layout 统一 —— token-major prefill 与 CPU 参考逐元素对比。
// 覆盖 S>1, H>1（GQA Hq=4/Hkv=2）：Q/K/V 均按 token-major 布局构造
//   q(s,h,d) = (s*Hq  + h )*D + d
//   k(s,kh,d)= (s*Hkv + kh)*D + d
// attention kernel 按同一契约读取；与 CPU 参考全维度逐元素比较，容差 1e-2f。
// ============================================================================
TEST_F(AttentionTest, PrefillLayoutMatchesCpuReference) {
    const int num_q_heads = 4;
    const int num_kv_heads = 2;
    const int seq_len = 16; // S > 1
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int group_size = num_q_heads / num_kv_heads;

    auto query = randomFP16(seq_len * num_q_heads * head_dim, 1.0f, 400);
    auto key = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 401);
    auto value = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 402);

    DeviceBuffer<half> d_query(seq_len * num_q_heads * head_dim);
    DeviceBuffer<half> d_key(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_value(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(seq_len * num_q_heads * head_dim);
    d_query.copyFromHost(query.data(), query.size());
    d_key.copyFromHost(key.data(), key.size());
    d_value.copyFromHost(value.data(), value.size());

    attention_prefill(d_query.data(), d_key.data(), d_value.data(), d_output.data(), scale,
                      num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();

    std::vector<half> output(seq_len * num_q_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    // CPU reference：causal（key_pos <= query_pos，注意是 <=），token-major 读取。
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_q_heads; ++h) {
            const int kh = h / group_size;
            std::vector<float> scores(static_cast<size_t>(s) + 1);
            float smax = -1e30f;
            for (int ks = 0; ks <= s; ++ks) {
                float acc = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    acc += __half2float(query[(s * num_q_heads + h) * head_dim + d]) *
                           __half2float(key[(ks * num_kv_heads + kh) * head_dim + d]);
                }
                scores[static_cast<size_t>(ks)] = acc * scale;
                smax = std::max(smax, scores[static_cast<size_t>(ks)]);
            }
            float sum_exp = 0.0f;
            for (int ks = 0; ks <= s; ++ks)
                sum_exp += std::exp(scores[static_cast<size_t>(ks)] - smax);
            for (int d = 0; d < head_dim; ++d) {
                float out_val = 0.0f;
                for (int ks = 0; ks <= s; ++ks) {
                    float w = std::exp(scores[static_cast<size_t>(ks)] - smax) / sum_exp;
                    out_val += w * __half2float(value[(ks * num_kv_heads + kh) * head_dim + d]);
                }
                const float actual = __half2float(output[(s * num_q_heads + h) * head_dim + d]);
                EXPECT_NEAR(actual, out_val, 1e-2f)
                    << "query_pos " << s << " head " << h << " dim " << d;
            }
        }
    }
}

// P1-6 回归：softmax 必须不受 48KB 动态共享内存默认上限约束。
// 旧实现按 (seq_len+32)*4B 申请动态共享内存，seq_len≈12K 即超限导致
// launch 失败（invalid argument），输出为垃圾值。O(1) 共享内存实现
// 应对任意 seq_len 正确完成。
TEST_F(AttentionTest, SoftmaxHandlesSeqLenBeyondSharedMemoryLimit) {
    int batch_size = 2;
    int seq_len = 20000; // (20000+32)*4B ≈ 80KB > 48KB 默认上限

    auto input = randomFP16(batch_size * seq_len, 2.0f, 777);

    DeviceBuffer<half> d_input(batch_size * seq_len);
    DeviceBuffer<half> d_output(batch_size * seq_len);
    d_input.copyFromHost(input.data(), input.size());

    EXPECT_NO_THROW({
        softmax(d_input.data(), d_output.data(), batch_size, seq_len);
        cudaError_t err = cudaDeviceSynchronize();
        EXPECT_EQ(err, cudaSuccess) << "softmax launch failed: "
                                    << cudaGetErrorString(err);
    });

    std::vector<half> output(batch_size * seq_len);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    // 每行和为 1，且最大 logit 位置对应最大概率
    for (int b = 0; b < batch_size; ++b) {
        float   sum = 0.0f;
        int     argmax_out = 0;
        int     argmax_in = 0;
        for (int i = 0; i < seq_len; ++i) {
            float v = __half2float(output[b * seq_len + i]);
            sum += v;
            if (v > __half2float(output[b * seq_len + argmax_out]))
                argmax_out = i;
            if (__half2float(input[b * seq_len + i]) >
                __half2float(input[b * seq_len + argmax_in]))
                argmax_in = i;
        }
        EXPECT_NEAR(sum, 1.0f, 0.01f) << "row " << b << " sum: " << sum;
        EXPECT_EQ(argmax_out, argmax_in) << "row " << b;
    }
}

TEST_F(AttentionTest, SoftmaxSumsToOne) {
    int batch_size = 4;
    int seq_len = 16;

    auto input = randomFP16(batch_size * seq_len, 2.0f);

    DeviceBuffer<half> d_input(batch_size * seq_len);
    DeviceBuffer<half> d_output(batch_size * seq_len);

    d_input.copyFromHost(input.data(), input.size());

    softmax(d_input.data(), d_output.data(), batch_size, seq_len);
    cudaDeviceSynchronize();

    std::vector<half> output(batch_size * seq_len);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    // Verify each row sums to approximately 1.0
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            sum += __half2float(output[b * seq_len + i]);
        }
        EXPECT_NEAR(sum, 1.0f, 0.01f) << "Row " << b << " sum: " << sum;
    }
}

#if 0
// Property-based tests
// Feature: tiny-llm-inference-engine, Property 3: Causal Masking Correctness
// Validates: Requirements 4.2
// NOTE: Disabled in CUDA translation units due to GCC 11/12 + nvcc
// compatibility issues with rapidcheck's GTest integration.

class AttentionPropertyTest : public AttentionTest {
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

// Property 3: Causal Masking Correctness
// For any attention computation at position t, the attention weights for
// positions > t must be exactly zero, ensuring no information leakage from
// future tokens.

RC_GTEST_FIXTURE_PROP(AttentionPropertyTest, CausalMaskZerosFuturePositions,
                      (int batch_raw, int heads_raw, int seq_raw, int dim_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions to reasonable ranges
    int   batch_size = 1 + (std::abs(batch_raw) % 4);
    int   num_heads = 1 + (std::abs(heads_raw) % 8);
    int   seq_len = 4 + (std::abs(seq_raw) % 60);   // 4 to 64
    int   head_dim = 16 + (std::abs(dim_raw) % 48); // 16 to 64
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Generate random Q and K
    auto query = randomFP16(batch_size * num_heads * seq_len * head_dim, 1.0f, seed);
    auto key = randomFP16(batch_size * num_heads * seq_len * head_dim, 1.0f, seed + 1);

    DeviceBuffer<half> d_query(batch_size * num_heads * seq_len * head_dim);
    DeviceBuffer<half> d_key(batch_size * num_heads * seq_len * head_dim);
    DeviceBuffer<half> d_weights(batch_size * num_heads * seq_len * seq_len);

    d_query.copyFromHost(query.data(), query.size());
    d_key.copyFromHost(key.data(), key.size());

    // Get attention weights with causal mask
    get_attention_weights(d_query.data(), d_key.data(), d_weights.data(), scale, batch_size,
                          num_heads, seq_len, seq_len, head_dim,
                          true); // apply_causal_mask = true

    cudaDeviceSynchronize();

    std::vector<half> weights(batch_size * num_heads * seq_len * seq_len);
    d_weights.copyToHost(weights.data(), weights.size());
    cudaDeviceSynchronize();

    // Property: For each query position t, weights for key positions > t must be
    // zero
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                for (int k_pos = q_pos + 1; k_pos < seq_len; ++k_pos) {
                    int   idx = ((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos;
                    float weight = __half2float(weights[idx]);

                    // Future positions must have zero weight
                    RC_ASSERT(weight == 0.0f);
                }
            }
        }
    }
}

RC_GTEST_FIXTURE_PROP(AttentionPropertyTest, CausalMaskAllowsPastPositions,
                      (int batch_raw, int heads_raw, int seq_raw, int dim_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions
    int   batch_size = 1 + (std::abs(batch_raw) % 4);
    int   num_heads = 1 + (std::abs(heads_raw) % 4);
    int   seq_len = 4 + (std::abs(seq_raw) % 28); // 4 to 32
    int   head_dim = 16 + (std::abs(dim_raw) % 48);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Generate random Q and K with non-zero values
    auto query = randomFP16(batch_size * num_heads * seq_len * head_dim, 1.0f, seed);
    auto key = randomFP16(batch_size * num_heads * seq_len * head_dim, 1.0f, seed + 1);

    DeviceBuffer<half> d_query(batch_size * num_heads * seq_len * head_dim);
    DeviceBuffer<half> d_key(batch_size * num_heads * seq_len * head_dim);
    DeviceBuffer<half> d_weights(batch_size * num_heads * seq_len * seq_len);

    d_query.copyFromHost(query.data(), query.size());
    d_key.copyFromHost(key.data(), key.size());

    // Get attention weights with causal mask
    get_attention_weights(d_query.data(), d_key.data(), d_weights.data(), scale, batch_size,
                          num_heads, seq_len, seq_len, head_dim, true);

    cudaDeviceSynchronize();

    std::vector<half> weights(batch_size * num_heads * seq_len * seq_len);
    d_weights.copyToHost(weights.data(), weights.size());
    cudaDeviceSynchronize();

    // Property: For each query position t, at least one weight for positions <= t
    // should be non-zero (unless all Q*K products happen to be exactly zero,
    // which is extremely unlikely)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                bool has_nonzero = false;
                for (int k_pos = 0; k_pos <= q_pos; ++k_pos) {
                    int   idx = ((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos;
                    float weight = __half2float(weights[idx]);
                    if (weight != 0.0f) {
                        has_nonzero = true;
                        break;
                    }
                }
                // Past/current positions should have non-zero weights
                RC_ASSERT(has_nonzero);
            }
        }
    }
}

RC_GTEST_FIXTURE_PROP(AttentionPropertyTest, SoftmaxOutputSumsToOne,
                      (int batch_raw, int seq_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions
    int batch_size = 1 + (std::abs(batch_raw) % 16);
    int seq_len = 4 + (std::abs(seq_raw) % 124); // 4 to 128

    // Generate random input
    auto input = randomFP16(batch_size * seq_len, 2.0f, seed);

    DeviceBuffer<half> d_input(batch_size * seq_len);
    DeviceBuffer<half> d_output(batch_size * seq_len);

    d_input.copyFromHost(input.data(), input.size());

    softmax(d_input.data(), d_output.data(), batch_size, seq_len);
    cudaDeviceSynchronize();

    std::vector<half> output(batch_size * seq_len);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    // Property: Each row should sum to approximately 1.0
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float val = __half2float(output[b * seq_len + i]);
            RC_ASSERT(val >= 0.0f); // Softmax outputs should be non-negative
            sum += val;
        }
        // Sum should be close to 1.0 (within FP16 tolerance)
        RC_ASSERT(sum > 0.98f && sum < 1.02f);
    }
}

RC_GTEST_FIXTURE_PROP(AttentionPropertyTest, SoftmaxPreservesOrder,
                      (int batch_raw, int seq_raw, unsigned seed)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions
    int batch_size = 1 + (std::abs(batch_raw) % 8);
    int seq_len = 4 + (std::abs(seq_raw) % 60);

    // Generate random input
    auto input = randomFP16(batch_size * seq_len, 2.0f, seed);

    DeviceBuffer<half> d_input(batch_size * seq_len);
    DeviceBuffer<half> d_output(batch_size * seq_len);

    d_input.copyFromHost(input.data(), input.size());

    softmax(d_input.data(), d_output.data(), batch_size, seq_len);
    cudaDeviceSynchronize();

    std::vector<half> output(batch_size * seq_len);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    // Property: Softmax should preserve relative ordering
    // If input[i] > input[j], then output[i] > output[j]
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = i + 1; j < seq_len; ++j) {
                float in_i = __half2float(input[b * seq_len + i]);
                float in_j = __half2float(input[b * seq_len + j]);
                float out_i = __half2float(output[b * seq_len + i]);
                float out_j = __half2float(output[b * seq_len + j]);

                // If inputs differ significantly, outputs should maintain order
                if (in_i > in_j + 0.1f) {
                    RC_ASSERT(out_i >= out_j);
                } else if (in_j > in_i + 0.1f) {
                    RC_ASSERT(out_j >= out_i);
                }
            }
        }
    }
}
#endif

// 任务 A2：GQA 映射的精确规格测试 —— Hq=4, Hkv=2, seq_len=8, head_dim=64。
// 与 CPU 参考（手动实现 kv_head = q_head / group_size 映射）逐元素比较，容差 1e-2f。
TEST_F(AttentionTest, GQAMappingDecodeMatchesCpuReference) {
    const int num_q_heads = 4;
    const int num_kv_heads = 2; // group_size = 2
    const int seq_len = 8;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int group_size = num_q_heads / num_kv_heads;

    auto query = randomFP16(num_q_heads * head_dim, 1.0f, 550);
    auto k_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 551);
    auto v_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 552);

    DeviceBuffer<half> d_query(num_q_heads * head_dim);
    DeviceBuffer<half> d_k_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_v_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(num_q_heads * head_dim);
    d_query.copyFromHost(query.data(), query.size());
    d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
    d_v_cache.copyFromHost(v_cache.data(), v_cache.size());

    attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(), d_output.data(), scale,
                     num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();

    std::vector<half> output(num_q_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    // CPU 参考：手动实现 GQA 映射（kv_head = q_head / group_size）
    for (int h = 0; h < num_q_heads; ++h) {
        int kh = h / group_size;
        std::vector<float> scores(seq_len);
        float smax = -1e30f;
        for (int s = 0; s < seq_len; ++s) {
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                acc += __half2float(query[h * head_dim + d]) *
                       __half2float(k_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            scores[s] = acc * scale;
            smax = std::max(smax, scores[s]);
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; ++s) sum_exp += std::exp(scores[s] - smax);
        for (int d = 0; d < head_dim; ++d) {
            float out_val = 0.0f;
            for (int s = 0; s < seq_len; ++s) {
                float w = std::exp(scores[s] - smax) / sum_exp;
                out_val += w * __half2float(v_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            EXPECT_NEAR(__half2float(output[h * head_dim + d]), out_val, 1e-2f)
                << "head " << h << " (kv_head=" << kh << ") dim " << d;
        }
    }
}

// ============================================================================
// Attention 数值验证（与 CPU 参考实现逐元素对比）
// 现有 AttentionTest 只验证"不 crash / 非零"，从未验证数值正确性。
// ============================================================================
TEST_F(AttentionTest, GQADecodeMatchesCpuReference) {
    const int num_q_heads = 4;
    const int num_kv_heads = 2;
    const int seq_len = 8;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int group_size = num_q_heads / num_kv_heads;

    auto query = randomFP16(num_q_heads * head_dim, 1.0f, 500);
    auto k_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 501);
    auto v_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 502);

    DeviceBuffer<half> d_query(num_q_heads * head_dim);
    DeviceBuffer<half> d_k_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_v_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(num_q_heads * head_dim);
    d_query.copyFromHost(query.data(), query.size());
    d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
    d_v_cache.copyFromHost(v_cache.data(), v_cache.size());

    attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(), d_output.data(), scale,
                     num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();

    std::vector<half> output(num_q_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    // CPU reference
    for (int h = 0; h < num_q_heads; ++h) {
        int kh = h / group_size;
        // scores
        std::vector<float> scores(seq_len);
        float smax = -1e30f;
        for (int s = 0; s < seq_len; ++s) {
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                acc += __half2float(query[h * head_dim + d]) *
                       __half2float(k_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            scores[s] = acc * scale;
            smax = std::max(smax, scores[s]);
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; ++s) sum_exp += std::exp(scores[s] - smax);
        for (int d = 0; d < head_dim; ++d) {
            float out_val = 0.0f;
            for (int s = 0; s < seq_len; ++s) {
                float w = std::exp(scores[s] - smax) / sum_exp;
                out_val += w * __half2float(v_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            EXPECT_NEAR(__half2float(output[h * head_dim + d]), out_val, 5e-2f)
                << "head " << h << " dim " << d;
        }
    }
}

// Long-sequence decode: spans many online-softmax tiles and exercises the
// re-scaled running max/sum path (seq_len > several ATTEND_TILE tiles).
TEST_F(AttentionTest, LongSequenceDecodeMatchesCpuReference) {
    const int num_q_heads = 4;
    const int num_kv_heads = 2;
    const int seq_len = 3000;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int group_size = num_q_heads / num_kv_heads;

    auto query = randomFP16(num_q_heads * head_dim, 1.0f, 600);
    auto k_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 601);
    auto v_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 602);

    DeviceBuffer<half> d_query(num_q_heads * head_dim);
    DeviceBuffer<half> d_k_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_v_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(num_q_heads * head_dim);
    d_query.copyFromHost(query.data(), query.size());
    d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
    d_v_cache.copyFromHost(v_cache.data(), v_cache.size());

    attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(), d_output.data(), scale,
                     num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();

    std::vector<half> output(num_q_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    // CPU reference (only compare few dims per head to keep test fast)
    for (int h = 0; h < num_q_heads; ++h) {
        int kh = h / group_size;
        std::vector<float> scores(seq_len);
        float smax = -1e30f;
        for (int s = 0; s < seq_len; ++s) {
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                acc += __half2float(query[h * head_dim + d]) *
                       __half2float(k_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            scores[s] = acc * scale;
            smax = std::max(smax, scores[s]);
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; ++s) sum_exp += std::exp(scores[s] - smax);
        for (int d = 0; d < head_dim; ++d) {
            float out_val = 0.0f;
            for (int s = 0; s < seq_len; ++s) {
                float w = std::exp(scores[s] - smax) / sum_exp;
                out_val += w * __half2float(v_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            float actual = __half2float(output[h * head_dim + d]);
            EXPECT_NEAR(actual, out_val, 8e-2f)
                << "head " << h << " dim " << d;
        }
    }
}

// ============================================================================
// Prefill 长序列 + 非整 tile 边界（1025 = 8×128 + 1）：覆盖 causal mask 与
// online-softmax 在 seq_len % ATTEND_TILE != 0 时的重缩放路径。
// ============================================================================
TEST_F(AttentionTest, LongSequencePrefillMatchesCpuReference) {
    const int num_q_heads = 4;
    const int num_kv_heads = 2;
    const int seq_len = 1025;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int group_size = num_q_heads / num_kv_heads;

    auto query = randomFP16(seq_len * num_q_heads * head_dim, 1.0f, 700);
    auto key = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 701);
    auto value = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 702);

    DeviceBuffer<half> d_query(seq_len * num_q_heads * head_dim);
    DeviceBuffer<half> d_key(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_value(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(seq_len * num_q_heads * head_dim);
    d_query.copyFromHost(query.data(), query.size());
    d_key.copyFromHost(key.data(), key.size());
    d_value.copyFromHost(value.data(), value.size());

    attention_prefill(d_query.data(), d_key.data(), d_value.data(), d_output.data(), scale,
                      num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();

    std::vector<half> output(seq_len * num_q_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    // 全维度数值比较只做边界 query_pos（含 tile 边界两侧与序列尾部），
    // 其余 query 只检查非零，避免 O(seq_len²×D×H) 的 CPU 参考拖慢单测。
    const int checked_pos[] = {0, 1, 127, 128, 129, 1024};
    const size_t num_checked = sizeof(checked_pos) / sizeof(checked_pos[0]);

    std::vector<bool> full_check(static_cast<size_t>(seq_len), false);
    for (size_t i = 0; i < num_checked; ++i) full_check[static_cast<size_t>(checked_pos[i])] = true;

    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_q_heads; ++h) {
            int kh = h / group_size;
            if (full_check[static_cast<size_t>(s)]) {
                // 只对 key_pos <= query_pos 求 score（causal，注意是 <=）
                std::vector<float> scores(static_cast<size_t>(s) + 1);
                float smax = -1e30f;
                for (int ks = 0; ks <= s; ++ks) {
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        acc += __half2float(query[(s * num_q_heads + h) * head_dim + d]) *
                               __half2float(key[(ks * num_kv_heads + kh) * head_dim + d]);
                    }
                    scores[static_cast<size_t>(ks)] = acc * scale;
                    smax = std::max(smax, scores[static_cast<size_t>(ks)]);
                }
                float sum_exp = 0.0f;
                for (int ks = 0; ks <= s; ++ks)
                    sum_exp += std::exp(scores[static_cast<size_t>(ks)] - smax);
                for (int d = 0; d < head_dim; ++d) {
                    float out_val = 0.0f;
                    for (int ks = 0; ks <= s; ++ks) {
                        float w = std::exp(scores[static_cast<size_t>(ks)] - smax) / sum_exp;
                        out_val += w *
                                   __half2float(value[(ks * num_kv_heads + kh) * head_dim + d]);
                    }
                    float actual = __half2float(output[(s * num_q_heads + h) * head_dim + d]);
                    EXPECT_NEAR(actual, out_val, 8e-2f)
                        << "query_pos " << s << " head " << h << " dim " << d;
                }
            } else {
                bool has_nonzero = false;
                for (int d = 0; d < head_dim; ++d) {
                    if (__half2float(output[(s * num_q_heads + h) * head_dim + d]) != 0.0f) {
                        has_nonzero = true;
                        break;
                    }
                }
                EXPECT_TRUE(has_nonzero) << "query_pos " << s << " head " << h << " all zero";
            }
        }
    }
}

// ============================================================================
// 任务 4.2：第二个真实模型的 GQA 配置映射验证（kernel 级）。
// 手头没有第二个 GGUF 时，先验证推荐模型配置的 group_size 映射：
//   - Llama-3.2-1B-Instruct：Hq=32, Hkv=8（group_size=4，与 Qwen 14→2 差异大）
//   - MQA 类模型：Hq=16, Hkv=1（group_size=16）
// 真实模型到位后，用 gated 测试（TLLM_GGUF_TEST_MODEL_2）做端到端验证。
// ============================================================================

// Llama-3.2-1B 的 GQA 比例：32 个 Q head 共享 8 个 KV head（group_size=4）。
TEST_F(AttentionTest, Llama32GQA32To8DecodeMatchesCpuReference) {
    const int num_q_heads = 32;
    const int num_kv_heads = 8;
    const int seq_len = 16;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int group_size = num_q_heads / num_kv_heads;

    auto query = randomFP16(num_q_heads * head_dim, 1.0f, 800);
    auto k_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 801);
    auto v_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 802);

    DeviceBuffer<half> d_query(num_q_heads * head_dim);
    DeviceBuffer<half> d_k_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_v_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(num_q_heads * head_dim);
    d_query.copyFromHost(query.data(), query.size());
    d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
    d_v_cache.copyFromHost(v_cache.data(), v_cache.size());

    attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(), d_output.data(), scale,
                     num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();

    std::vector<half> output(num_q_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    for (int h = 0; h < num_q_heads; ++h) {
        int kh = h / group_size;
        std::vector<float> scores(seq_len);
        float smax = -1e30f;
        for (int s = 0; s < seq_len; ++s) {
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                acc += __half2float(query[h * head_dim + d]) *
                       __half2float(k_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            scores[s] = acc * scale;
            smax = std::max(smax, scores[s]);
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; ++s) sum_exp += std::exp(scores[s] - smax);
        for (int d = 0; d < head_dim; ++d) {
            float out_val = 0.0f;
            for (int s = 0; s < seq_len; ++s) {
                float w = std::exp(scores[s] - smax) / sum_exp;
                out_val += w * __half2float(v_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            EXPECT_NEAR(__half2float(output[h * head_dim + d]), out_val, 8e-2f)
                << "head " << h << " dim " << d;
        }
    }
}

// MQA：16 个 Q head 共享 1 个 KV head（group_size=16）。
TEST_F(AttentionTest, MQA16To1DecodeMatchesCpuReference) {
    const int num_q_heads = 16;
    const int num_kv_heads = 1;
    const int seq_len = 16;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int group_size = num_q_heads / num_kv_heads;

    auto query = randomFP16(num_q_heads * head_dim, 1.0f, 900);
    auto k_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 901);
    auto v_cache = randomFP16(seq_len * num_kv_heads * head_dim, 1.0f, 902);

    DeviceBuffer<half> d_query(num_q_heads * head_dim);
    DeviceBuffer<half> d_k_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_v_cache(seq_len * num_kv_heads * head_dim);
    DeviceBuffer<half> d_output(num_q_heads * head_dim);
    d_query.copyFromHost(query.data(), query.size());
    d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
    d_v_cache.copyFromHost(v_cache.data(), v_cache.size());

    attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(), d_output.data(), scale,
                     num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();

    std::vector<half> output(num_q_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();

    for (int h = 0; h < num_q_heads; ++h) {
        int kh = h / group_size;
        std::vector<float> scores(seq_len);
        float smax = -1e30f;
        for (int s = 0; s < seq_len; ++s) {
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                acc += __half2float(query[h * head_dim + d]) *
                       __half2float(k_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            scores[s] = acc * scale;
            smax = std::max(smax, scores[s]);
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; ++s) sum_exp += std::exp(scores[s] - smax);
        for (int d = 0; d < head_dim; ++d) {
            float out_val = 0.0f;
            for (int s = 0; s < seq_len; ++s) {
                float w = std::exp(scores[s] - smax) / sum_exp;
                out_val += w * __half2float(v_cache[(s * num_kv_heads + kh) * head_dim + d]);
            }
            EXPECT_NEAR(__half2float(output[h * head_dim + d]), out_val, 8e-2f)
                << "head " << h << " dim " << d;
        }
    }
}

// ============================================================================
// 任务 A3：RoPE 进入计算路径 —— apply_rope_inplace 与 CPU 参考（half-split
// rotate_half 语义）逐元素比较，容差 1e-3f。
//
// 语义（与 rope.cuh 文档一致）：
//   x1 = x[d] (d in [0, D/2)), x2 = x[d + D/2]
//   out[d]        = x1*cos(pos,d) - x2*sin(pos,d)
//   out[d + D/2]  = x1*sin(pos,d) + x2*cos(pos,d)
//   angle(pos,d)  = pos * theta^(-2d/D)
// ============================================================================
TEST(RoPETest, ApplyInplaceMatchesReference) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
    cudaSetDevice(0);

    const int num_q_heads = 4;
    const int num_kv_heads = 2;
    const int num_tokens = 3; // 覆盖多个位置（start_position + s）
    const int head_dim = 64;
    const int half_d = head_dim / 2;
    const int start_position = 5;
    const float theta = 10000.0f;
    const int max_seq_len = 32;

    // 预计算 cos/sin 表 [max_seq_len, D/2]
    std::vector<float> cos_cache(static_cast<size_t>(max_seq_len) * half_d);
    std::vector<float> sin_cache(static_cast<size_t>(max_seq_len) * half_d);
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int d = 0; d < half_d; ++d) {
            float freq = 1.0f / std::pow(theta, (2.0f * d) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * freq;
            cos_cache[static_cast<size_t>(pos) * half_d + d] = std::cos(angle);
            sin_cache[static_cast<size_t>(pos) * half_d + d] = std::sin(angle);
        }
    }

    // 随机 Q/K（token-major），范围 [-0.5, 0.5] 使输出落在 fp16 高精度区
    std::vector<half> q(static_cast<size_t>(num_tokens) * num_q_heads * head_dim);
    std::vector<half> k(static_cast<size_t>(num_tokens) * num_kv_heads * head_dim);
    {
        std::mt19937                          gen(100);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto &v : q) v = __float2half(dist(gen));
        for (auto &v : k) v = __float2half(dist(gen));
    }
    const std::vector<half> q_ref = q;
    const std::vector<half> k_ref = k;

    DeviceBuffer<half>  d_q(num_tokens * num_q_heads * head_dim);
    DeviceBuffer<half>  d_k(num_tokens * num_kv_heads * head_dim);
    DeviceBuffer<float> d_cos(max_seq_len * half_d);
    DeviceBuffer<float> d_sin(max_seq_len * half_d);
    d_q.copyFromHost(q.data(), q.size());
    d_k.copyFromHost(k.data(), k.size());
    d_cos.copyFromHost(cos_cache.data(), cos_cache.size());
    d_sin.copyFromHost(sin_cache.data(), sin_cache.size());

    kernels::apply_rope_inplace(d_q.data(), d_k.data(), d_cos.data(), d_sin.data(), num_tokens,
                                start_position, num_q_heads, num_kv_heads, head_dim);
    cudaDeviceSynchronize();

    std::vector<half> q_out(static_cast<size_t>(num_tokens) * num_q_heads * head_dim);
    std::vector<half> k_out(static_cast<size_t>(num_tokens) * num_kv_heads * head_dim);
    d_q.copyToHost(q_out.data(), q_out.size());
    d_k.copyToHost(k_out.data(), k_out.size());
    cudaDeviceSynchronize();

    // CPU 参考：half-split 旋转，逐元素比较
    for (int s = 0; s < num_tokens; ++s) {
        const int pos = start_position + s;
        for (int h = 0; h < num_q_heads; ++h) {
            for (int d = 0; d < half_d; ++d) {
                const float c = cos_cache[static_cast<size_t>(pos) * half_d + d];
                const float sn = sin_cache[static_cast<size_t>(pos) * half_d + d];
                const float x1 = __half2float(q_ref[(s * num_q_heads + h) * head_dim + d]);
                const float x2 = __half2float(q_ref[(s * num_q_heads + h) * head_dim + d + half_d]);
                const float exp_first = x1 * c - x2 * sn;
                const float exp_second = x1 * sn + x2 * c;
                const float got_first = __half2float(q_out[(s * num_q_heads + h) * head_dim + d]);
                const float got_second =
                    __half2float(q_out[(s * num_q_heads + h) * head_dim + d + half_d]);
                EXPECT_NEAR(got_first, exp_first, 1e-3f)
                    << "Q s=" << s << " h=" << h << " d=" << d;
                EXPECT_NEAR(got_second, exp_second, 1e-3f)
                    << "Q s=" << s << " h=" << h << " d=" << d;
            }
        }
        for (int kh = 0; kh < num_kv_heads; ++kh) {
            for (int d = 0; d < half_d; ++d) {
                const float c = cos_cache[static_cast<size_t>(pos) * half_d + d];
                const float sn = sin_cache[static_cast<size_t>(pos) * half_d + d];
                const float x1 = __half2float(k_ref[(s * num_kv_heads + kh) * head_dim + d]);
                const float x2 = __half2float(k_ref[(s * num_kv_heads + kh) * head_dim + d + half_d]);
                const float exp_first = x1 * c - x2 * sn;
                const float exp_second = x1 * sn + x2 * c;
                const float got_first = __half2float(k_out[(s * num_kv_heads + kh) * head_dim + d]);
                const float got_second =
                    __half2float(k_out[(s * num_kv_heads + kh) * head_dim + d + half_d]);
                EXPECT_NEAR(got_first, exp_first, 1e-3f)
                    << "K s=" << s << " kh=" << kh << " d=" << d;
                EXPECT_NEAR(got_second, exp_second, 1e-3f)
                    << "K s=" << s << " kh=" << kh << " d=" << d;
            }
        }
    }
}

// ── 分页 KV：scatter/gather 原语 ────────────────────────────────
// 第一版简单正确即可（一个元素一个线程），正确性优先、不做优化。

namespace {

// 把连续 src（token-major）按块表散布到 pool 的 CPU 参考实现。
// 返回目标 pool 索引（以 half 元素为单位）处的期望值映射（仅用于校验）。
std::vector<half> cpuScatter(const std::vector<half> &src, const std::vector<int> &block_table,
                             int num_tokens, int position, int block_size, int chunk_dim,
                             int pool_blocks) {
    std::vector<half> pool(static_cast<size_t>(pool_blocks) * block_size * chunk_dim,
                           __float2half(0.0f));
    for (int t = 0; t < num_tokens; ++t) {
        int abs = position + t;
        int b = abs / block_size;
        int within = abs - b * block_size;
        int block_id = block_table[b];
        for (int c = 0; c < chunk_dim; ++c) {
            size_t dst = ((size_t)block_id * block_size + within) * chunk_dim + c;
            pool[dst] = src[(size_t)t * chunk_dim + c];
        }
    }
    return pool;
}

// 从 pool 按块表读回前 visible_tokens 行的 CPU 参考。
std::vector<half> cpuGather(const std::vector<half> &pool, const std::vector<int> &block_table,
                            int visible_tokens, int block_size, int chunk_dim) {
    std::vector<half> dst(static_cast<size_t>(visible_tokens) * chunk_dim);
    for (int t = 0; t < visible_tokens; ++t) {
        int b = t / block_size;
        int within = t - b * block_size;
        int block_id = block_table[b];
        for (int c = 0; c < chunk_dim; ++c) {
            size_t src = ((size_t)block_id * block_size + within) * chunk_dim + c;
            dst[(size_t)t * chunk_dim + c] = pool[src];
        }
    }
    return dst;
}

std::vector<half> randomFp16(size_t n, unsigned seed) {
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<half>                     v(n);
    for (auto &x : v) x = __float2half(dist(gen));
    return v;
}

} // namespace

TEST(PagedKvTest, PagedScatterGatherRoundTrip) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
    const int block_size = 16;
    const int chunk_dim = 128;
    const int num_tokens = 17; // 跨块边界：block 0 的 16 行 + block 1 的 1 行
    const int pool_blocks = 8;
    const std::vector<int> block_table = {3, 7};

    const std::vector<half> src = randomFp16(static_cast<size_t>(num_tokens) * chunk_dim, 7);
    DeviceBuffer<half> d_src(src.size());
    d_src.copyFromHost(src.data(), src.size());

    DeviceBuffer<half>  d_pool(static_cast<size_t>(pool_blocks) * block_size * chunk_dim);
    DeviceBuffer<half>  d_dst(src.size());
    DeviceBuffer<int>   d_table(block_table.size());
    d_table.copyFromHost(block_table.data(), block_table.size());

    kernels::paged_scatter_blocks(d_src.data(), d_pool.data(), d_table.data(), num_tokens,
                                  /*position=*/0, block_size, chunk_dim, pool_blocks);
    kernels::paged_gather_blocks(d_dst.data(), d_pool.data(), d_table.data(), num_tokens,
                                 block_size, chunk_dim, pool_blocks);
    cudaDeviceSynchronize();

    std::vector<half> got(src.size());
    d_dst.copyToHost(got.data(), got.size());
    cudaDeviceSynchronize();
    for (size_t i = 0; i < src.size(); ++i) {
        EXPECT_NEAR(__half2float(got[i]), __half2float(src[i]), 1e-2f)
            << "round-trip element " << i << " differs";
    }
}

TEST(PagedKvTest, PagedGatherPartialVisibility) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
    const int block_size = 16;
    const int chunk_dim = 64;
    const int num_tokens = 16;
    const int visible_tokens = 7; // 只读前 7 个位置
    const int pool_blocks = 8;
    const std::vector<int> block_table = {2};

    const std::vector<half> src = randomFp16(static_cast<size_t>(num_tokens) * chunk_dim, 11);
    DeviceBuffer<half> d_src(src.size());
    d_src.copyFromHost(src.data(), src.size());

    DeviceBuffer<half> d_pool(static_cast<size_t>(pool_blocks) * block_size * chunk_dim);
    DeviceBuffer<half> d_dst(static_cast<size_t>(visible_tokens) * chunk_dim);
    DeviceBuffer<int>  d_table(block_table.size());
    d_table.copyFromHost(block_table.data(), block_table.size());

    kernels::paged_scatter_blocks(d_src.data(), d_pool.data(), d_table.data(), num_tokens, 0,
                                  block_size, chunk_dim, pool_blocks);
    kernels::paged_gather_blocks(d_dst.data(), d_pool.data(), d_table.data(), visible_tokens,
                                 block_size, chunk_dim, pool_blocks);
    cudaDeviceSynchronize();

    // CPU 参考：直接取 src 前 visible_tokens 行
    std::vector<half> got(static_cast<size_t>(visible_tokens) * chunk_dim);
    d_dst.copyToHost(got.data(), got.size());
    cudaDeviceSynchronize();
    for (int t = 0; t < visible_tokens; ++t) {
        for (int c = 0; c < chunk_dim; ++c) {
            size_t i = (size_t)t * chunk_dim + c;
            EXPECT_NEAR(__half2float(got[i]), __half2float(src[i]), 1e-2f)
                << "visible token " << t << " dim " << c << " differs";
        }
    }
}

TEST(PagedKvTest, PagedScatterWritesAtAbsolutePosition) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
    const int block_size = 16;
    const int chunk_dim = 32;
    const int position = 20; // abs=20 → block 1（abs/16），块内 offset 4
    const int num_tokens = 1;
    const int pool_blocks = 8;
    const std::vector<int> block_table = {0, 5};

    const std::vector<half> src = randomFp16(static_cast<size_t>(num_tokens) * chunk_dim, 13);
    DeviceBuffer<half> d_src(src.size());
    d_src.copyFromHost(src.data(), src.size());

    DeviceBuffer<half> d_pool(static_cast<size_t>(pool_blocks) * block_size * chunk_dim);
    DeviceBuffer<int>  d_table(block_table.size());
    d_table.copyFromHost(block_table.data(), block_table.size());

    kernels::paged_scatter_blocks(d_src.data(), d_pool.data(), d_table.data(), num_tokens,
                                  position, block_size, chunk_dim, pool_blocks);
    cudaDeviceSynchronize();

    // 期望落在 block_table[1]=5 块的第 4 行（块内 offset = 20 - 1*16 = 4）
    std::vector<half> pool(static_cast<size_t>(pool_blocks) * block_size * chunk_dim);
    d_pool.copyToHost(pool.data(), pool.size());
    cudaDeviceSynchronize();
    const size_t expected_base = ((size_t)5 * block_size + 4) * chunk_dim;
    for (int c = 0; c < chunk_dim; ++c) {
        EXPECT_NEAR(__half2float(pool[expected_base + c]), __half2float(src[c]), 1e-2f)
            << "row 4 of block 5, dim " << c << " differs";
    }
}

// P2-19 回归：块表含越界 id 时，scatter 必须跳过写入、gather 写 0，
// 绝不能越界访存（illegal address 会毒化整个 CUDA 上下文）。
TEST(PagedKvTest, InvalidBlockIdIsGuardedNotDereferenced) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
    const int block_size = 16;
    const int chunk_dim = 64;
    const int num_tokens = 8;
    const int pool_blocks = 4;

    const std::vector<half> src = randomFp16(static_cast<size_t>(num_tokens) * chunk_dim, 31);
    DeviceBuffer<half> d_src(src.size());
    d_src.copyFromHost(src.data(), src.size());

    // 块表全部越界：负值与 == max_num_blocks
    const std::vector<int> bad_table = {-1, pool_blocks};
    DeviceBuffer<int>      d_table(bad_table.size());
    d_table.copyFromHost(bad_table.data(), bad_table.size());

    // scatter：pool 预填哨兵值，越界 id 不得改动任何槽位
    DeviceBuffer<half> d_pool(static_cast<size_t>(pool_blocks) * block_size * chunk_dim);
    const size_t pool_elems = static_cast<size_t>(pool_blocks) * block_size * chunk_dim;
    std::vector<half>  sentinel(pool_elems, __float2half(7.0f));
    d_pool.copyFromHost(sentinel.data(), sentinel.size());
    kernels::paged_scatter_blocks(d_src.data(), d_pool.data(), d_table.data(), num_tokens,
                                  /*position=*/0, block_size, chunk_dim, pool_blocks);
    cudaError_t err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "scatter with bad table must not fault: "
                                << cudaGetErrorString(err);
    std::vector<half> pool_after(pool_elems);
    d_pool.copyToHost(pool_after.data(), pool_after.size());
    for (size_t i = 0; i < pool_after.size(); ++i) {
        ASSERT_EQ(__half2float(pool_after[i]), 7.0f)
            << "pool slot " << i << " was written via invalid block id";
    }

    // gather：越界 id 对应行写 0，且不得触发 illegal address
    DeviceBuffer<half> d_dst(static_cast<size_t>(num_tokens) * chunk_dim);
    kernels::paged_gather_blocks(d_dst.data(), d_pool.data(), d_table.data(), num_tokens,
                                 block_size, chunk_dim, pool_blocks);
    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "gather with bad table must not fault: "
                                << cudaGetErrorString(err);
    std::vector<half> dst(num_tokens * chunk_dim);
    d_dst.copyToHost(dst.data(), dst.size());
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_EQ(__half2float(dst[i]), 0.0f) << "dst[" << i << "] should be zero-filled";
    }
}

TEST(PagedKvTest, PagedLayersDoNotOverlap) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
    const int block_size = 16;
    const int chunk_dim = 64;
    const int num_tokens = 8;
    const int pool_blocks = 4;
    const int num_layers = 2;
    const int layer_stride = pool_blocks * block_size * chunk_dim;

    // 两个"层"用不同 pool 偏移写不同数据
    const std::vector<half> src0 = randomFp16(static_cast<size_t>(num_tokens) * chunk_dim, 21);
    const std::vector<half> src1 = randomFp16(static_cast<size_t>(num_tokens) * chunk_dim, 22);
    DeviceBuffer<half> d_src0(src0.size());
    DeviceBuffer<half> d_src1(src1.size());
    d_src0.copyFromHost(src0.data(), src0.size());
    d_src1.copyFromHost(src1.data(), src1.size());

    DeviceBuffer<half> d_pool(static_cast<size_t>(num_layers) * layer_stride);
    const std::vector<int> block_table = {0};
    DeviceBuffer<int>      d_table(1);
    d_table.copyFromHost(block_table.data(), 1);

    // 层 0 写 block 0、层 1 写 block 0（不同 layer 偏移）
    kernels::paged_scatter_blocks(d_src0.data(), d_pool.data(), d_table.data(), num_tokens, 0,
                                  block_size, chunk_dim, pool_blocks);
    kernels::paged_scatter_blocks(d_src1.data(), d_pool.data() + layer_stride, d_table.data(),
                                  num_tokens, 0, block_size, chunk_dim, pool_blocks);
    cudaDeviceSynchronize();

    std::vector<half> pool(static_cast<size_t>(num_layers) * layer_stride);
    d_pool.copyToHost(pool.data(), pool.size());
    cudaDeviceSynchronize();

    // 层 0 区域 = src0，层 1 区域 = src1，且互不污染
    for (int t = 0; t < num_tokens; ++t) {
        for (int c = 0; c < chunk_dim; ++c) {
            size_t i = (size_t)t * chunk_dim + c;
            EXPECT_NEAR(__half2float(pool[i]), __half2float(src0[i]), 1e-2f)
                << "layer0 token " << t << " dim " << c << " differs";
            EXPECT_NEAR(__half2float(pool[layer_stride + i]), __half2float(src1[i]), 1e-2f)
                << "layer1 token " << t << " dim " << c << " differs";
        }
    }
}

// 修复验证：add_bias_inplace 在非 256 倍数尺寸下（尾块存在越界线程）结果正确
// 且不越界。修复前 kernel 无条件读写 data[idx]，尾块 idx >= rows*cols 的
// 线程是未定义行为。
TEST(ElementwiseTest, AddBiasNonAlignedSizeCorrect) {
    if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
    cudaSetDevice(0);

    const int rows = 1, cols = 896; // total = 896 不是 256 倍数
    std::vector<half> data(static_cast<size_t>(rows) * cols, __float2half(1.0f));
    std::vector<half> bias(cols, __float2half(0.5f));

    DeviceBuffer<half> d_data(static_cast<size_t>(rows) * cols);
    DeviceBuffer<half> d_bias(cols);
    d_data.copyFromHost(data.data(), data.size());
    d_bias.copyFromHost(bias.data(), bias.size());

    add_bias_inplace(d_data.data(), d_bias.data(), rows, cols);
    cudaDeviceSynchronize();

    std::vector<half> out(static_cast<size_t>(rows) * cols);
    d_data.copyToHost(out.data(), out.size());
    cudaDeviceSynchronize();

    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(__half2float(out[i]), 1.5f, 1e-6f) << "index " << i;
    }
}
