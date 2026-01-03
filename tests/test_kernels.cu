#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <random>
#include <cmath>
#include "rmsnorm.cuh"
#include "tiny_llm/cuda_utils.h"

using namespace tiny_llm;
using namespace tiny_llm::kernels;

// Helper class for GPU test fixtures
class RMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // Generate random FP16 vector
    std::vector<half> randomFP16(int size, float scale = 1.0f, unsigned seed = 42) {
        std::vector<half> data(size);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto& v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }
    
    // Generate FP16 weights (positive values for RMSNorm weights)
    std::vector<half> randomWeights(int size, unsigned seed = 123) {
        std::vector<half> data(size);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(0.5f, 1.5f);
        for (auto& v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }
    
    // Compute RMS of a vector (on host)
    float computeRMS(const std::vector<half>& data) {
        float sum_sq = 0.0f;
        for (const auto& v : data) {
            float val = __half2float(v);
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq / data.size());
    }
    
    // Compute RMS for a specific row in a batch
    float computeRowRMS(const std::vector<half>& data, int row, int hidden_dim) {
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
    int batch_size = 2;
    int hidden_dim = 64;
    float eps = 1e-5f;
    
    auto input = randomFP16(batch_size * hidden_dim, 1.0f);
    auto weight = randomWeights(hidden_dim);
    
    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);
    
    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);
    
    rmsnorm(d_input.data(), d_weight.data(), d_output.data(),
            batch_size, hidden_dim, eps);
    
    cudaDeviceSynchronize();
    
    std::vector<half> output(batch_size * hidden_dim);
    d_output.copyToHost(output.data(), batch_size * hidden_dim);
    cudaDeviceSynchronize();
    
    // Verify output is not all zeros
    bool has_nonzero = false;
    for (const auto& v : output) {
        if (__half2float(v) != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(RMSNormTest, SmallHiddenDim) {
    int batch_size = 4;
    int hidden_dim = 32;
    float eps = 1e-5f;
    
    auto input = randomFP16(batch_size * hidden_dim, 1.0f);
    auto weight = randomWeights(hidden_dim);
    
    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);
    
    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);
    
    EXPECT_NO_THROW({
        rmsnorm(d_input.data(), d_weight.data(), d_output.data(),
                batch_size, hidden_dim, eps);
        cudaDeviceSynchronize();
    });
}

TEST_F(RMSNormTest, LargeHiddenDim) {
    int batch_size = 2;
    int hidden_dim = 4096;
    float eps = 1e-5f;
    
    auto input = randomFP16(batch_size * hidden_dim, 1.0f);
    auto weight = randomWeights(hidden_dim);
    
    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);
    
    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);
    
    EXPECT_NO_THROW({
        rmsnorm(d_input.data(), d_weight.data(), d_output.data(),
                batch_size, hidden_dim, eps);
        cudaDeviceSynchronize();
    });
}

TEST_F(RMSNormTest, InPlaceVersion) {
    int batch_size = 2;
    int hidden_dim = 128;
    float eps = 1e-5f;
    
    auto input = randomFP16(batch_size * hidden_dim, 1.0f);
    auto weight = randomWeights(hidden_dim);
    
    DeviceBuffer<half> d_x(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    
    d_x.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(weight.data(), hidden_dim);
    
    EXPECT_NO_THROW({
        rmsnorm_inplace(d_x.data(), d_weight.data(),
                        batch_size, hidden_dim, eps);
        cudaDeviceSynchronize();
    });
}


// Property-based tests
// Feature: tiny-llm-inference-engine, Property 4: RMSNorm Output Properties
// Validates: Requirements 4.4

class RMSNormPropertyTest : public RMSNormTest {};

// Property 4: RMSNorm Output Properties
// For any input tensor x, the RMSNorm output y (before weight multiplication) should satisfy:
// sqrt(mean(y^2)) ≈ 1.0 (within floating point tolerance)
//
// Since the kernel applies weight multiplication, we test with unit weights to verify
// the normalization property, then test with random weights to verify weight application.

RC_GTEST_FIXTURE_PROP(RMSNormPropertyTest, OutputRMSIsOne,
                      (int batch_raw, int dim_raw, unsigned seed)) {
    // Constrain dimensions to reasonable ranges
    int batch_size = 1 + (std::abs(batch_raw) % 16);
    int hidden_dim = 32 + (std::abs(dim_raw) % 2016);  // 32 to 2048
    float eps = 1e-5f;
    
    // Generate random input
    auto input = randomFP16(batch_size * hidden_dim, 2.0f, seed);
    
    // Use unit weights to test pure normalization
    std::vector<half> unit_weight(hidden_dim);
    for (auto& w : unit_weight) {
        w = __float2half(1.0f);
    }
    
    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);
    
    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(unit_weight.data(), hidden_dim);
    
    rmsnorm(d_input.data(), d_weight.data(), d_output.data(),
            batch_size, hidden_dim, eps);
    
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
    // Constrain dimensions
    int batch_size = 1 + (std::abs(batch_raw) % 8);
    int hidden_dim = 64 + (std::abs(dim_raw) % 448);  // 64 to 512
    float eps = 1e-5f;
    
    // Generate random input
    auto input = randomFP16(batch_size * hidden_dim, 1.0f, seed);
    
    // Test with constant weight = 2.0
    std::vector<half> const_weight(hidden_dim);
    float weight_val = 2.0f;
    for (auto& w : const_weight) {
        w = __float2half(weight_val);
    }
    
    DeviceBuffer<half> d_input(batch_size * hidden_dim);
    DeviceBuffer<half> d_weight(hidden_dim);
    DeviceBuffer<half> d_output(batch_size * hidden_dim);
    
    d_input.copyFromHost(input.data(), batch_size * hidden_dim);
    d_weight.copyFromHost(const_weight.data(), hidden_dim);
    
    rmsnorm(d_input.data(), d_weight.data(), d_output.data(),
            batch_size, hidden_dim, eps);
    
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
    // Constrain dimensions
    int batch_size = 1 + (std::abs(batch_raw) % 8);
    int hidden_dim = 32 + (std::abs(dim_raw) % 480);
    float eps = 1e-5f;
    
    // Generate non-zero input
    auto input = randomFP16(batch_size * hidden_dim, 1.0f, seed);
    auto weight = randomWeights(hidden_dim, seed + 1);
    
    // Ensure input is not all zeros
    bool input_has_nonzero = false;
    for (const auto& v : input) {
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
    
    rmsnorm(d_input.data(), d_weight.data(), d_output.data(),
            batch_size, hidden_dim, eps);
    
    cudaDeviceSynchronize();
    
    std::vector<half> output(batch_size * hidden_dim);
    d_output.copyToHost(output.data(), batch_size * hidden_dim);
    cudaDeviceSynchronize();
    
    // Property: Output should have non-zero values
    bool has_nonzero = false;
    for (const auto& v : output) {
        if (std::abs(__half2float(v)) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    RC_ASSERT(has_nonzero);
}

RC_GTEST_FIXTURE_PROP(RMSNormPropertyTest, InPlaceEquivalence,
                      (int batch_raw, int dim_raw, unsigned seed)) {
    // Constrain dimensions
    int batch_size = 1 + (std::abs(batch_raw) % 8);
    int hidden_dim = 64 + (std::abs(dim_raw) % 448);
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
    
    rmsnorm(d_input1.data(), d_weight.data(), d_output.data(),
            batch_size, hidden_dim, eps);
    
    // Run in-place version
    DeviceBuffer<half> d_input2(batch_size * hidden_dim);
    d_input2.copyFromHost(input.data(), batch_size * hidden_dim);
    
    rmsnorm_inplace(d_input2.data(), d_weight.data(),
                    batch_size, hidden_dim, eps);
    
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


// ============================================================================
// Attention Kernel Tests
// ============================================================================

#include "attention.cuh"

class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // Generate random FP16 tensor
    std::vector<half> randomFP16(int size, float scale = 1.0f, unsigned seed = 42) {
        std::vector<half> data(size);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto& v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }
};

// Unit tests for Attention
TEST_F(AttentionTest, BasicDecodeAttention) {
    int batch_size = 1;
    int num_heads = 2;
    int seq_len = 8;
    int head_dim = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    auto query = randomFP16(batch_size * num_heads * head_dim);
    auto k_cache = randomFP16(batch_size * num_heads * seq_len * head_dim);
    auto v_cache = randomFP16(batch_size * num_heads * seq_len * head_dim);
    
    DeviceBuffer<half> d_query(batch_size * num_heads * head_dim);
    DeviceBuffer<half> d_k_cache(batch_size * num_heads * seq_len * head_dim);
    DeviceBuffer<half> d_v_cache(batch_size * num_heads * seq_len * head_dim);
    DeviceBuffer<half> d_output(batch_size * num_heads * head_dim);
    
    d_query.copyFromHost(query.data(), query.size());
    d_k_cache.copyFromHost(k_cache.data(), k_cache.size());
    d_v_cache.copyFromHost(v_cache.data(), v_cache.size());
    
    EXPECT_NO_THROW({
        attention_decode(d_query.data(), d_k_cache.data(), d_v_cache.data(),
                        d_output.data(), scale, batch_size, num_heads, seq_len, head_dim);
        cudaDeviceSynchronize();
    });
    
    std::vector<half> output(batch_size * num_heads * head_dim);
    d_output.copyToHost(output.data(), output.size());
    cudaDeviceSynchronize();
    
    // Verify output is not all zeros
    bool has_nonzero = false;
    for (const auto& v : output) {
        if (__half2float(v) != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(AttentionTest, BasicPrefillAttention) {
    int batch_size = 1;
    int num_heads = 2;
    int seq_len = 8;
    int head_dim = 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    auto query = randomFP16(batch_size * num_heads * seq_len * head_dim);
    auto key = randomFP16(batch_size * num_heads * seq_len * head_dim);
    auto value = randomFP16(batch_size * num_heads * seq_len * head_dim);
    
    DeviceBuffer<half> d_query(batch_size * num_heads * seq_len * head_dim);
    DeviceBuffer<half> d_key(batch_size * num_heads * seq_len * head_dim);
    DeviceBuffer<half> d_value(batch_size * num_heads * seq_len * head_dim);
    DeviceBuffer<half> d_output(batch_size * num_heads * seq_len * head_dim);
    
    d_query.copyFromHost(query.data(), query.size());
    d_key.copyFromHost(key.data(), key.size());
    d_value.copyFromHost(value.data(), value.size());
    
    EXPECT_NO_THROW({
        attention_prefill(d_query.data(), d_key.data(), d_value.data(),
                         d_output.data(), scale, batch_size, num_heads, seq_len, head_dim);
        cudaDeviceSynchronize();
    });
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

// Property-based tests
// Feature: tiny-llm-inference-engine, Property 3: Causal Masking Correctness
// Validates: Requirements 4.2

class AttentionPropertyTest : public AttentionTest {};

// Property 3: Causal Masking Correctness
// For any attention computation at position t, the attention weights for positions > t
// must be exactly zero, ensuring no information leakage from future tokens.

RC_GTEST_FIXTURE_PROP(AttentionPropertyTest, CausalMaskZerosFuturePositions,
                      (int batch_raw, int heads_raw, int seq_raw, int dim_raw, unsigned seed)) {
    // Constrain dimensions to reasonable ranges
    int batch_size = 1 + (std::abs(batch_raw) % 4);
    int num_heads = 1 + (std::abs(heads_raw) % 8);
    int seq_len = 4 + (std::abs(seq_raw) % 60);  // 4 to 64
    int head_dim = 16 + (std::abs(dim_raw) % 48);  // 16 to 64
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
    get_attention_weights(d_query.data(), d_key.data(), d_weights.data(),
                         scale, batch_size, num_heads, seq_len, seq_len, head_dim,
                         true);  // apply_causal_mask = true
    
    cudaDeviceSynchronize();
    
    std::vector<half> weights(batch_size * num_heads * seq_len * seq_len);
    d_weights.copyToHost(weights.data(), weights.size());
    cudaDeviceSynchronize();
    
    // Property: For each query position t, weights for key positions > t must be zero
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                for (int k_pos = q_pos + 1; k_pos < seq_len; ++k_pos) {
                    int idx = ((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos;
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
    // Constrain dimensions
    int batch_size = 1 + (std::abs(batch_raw) % 4);
    int num_heads = 1 + (std::abs(heads_raw) % 4);
    int seq_len = 4 + (std::abs(seq_raw) % 28);  // 4 to 32
    int head_dim = 16 + (std::abs(dim_raw) % 48);
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
    get_attention_weights(d_query.data(), d_key.data(), d_weights.data(),
                         scale, batch_size, num_heads, seq_len, seq_len, head_dim,
                         true);
    
    cudaDeviceSynchronize();
    
    std::vector<half> weights(batch_size * num_heads * seq_len * seq_len);
    d_weights.copyToHost(weights.data(), weights.size());
    cudaDeviceSynchronize();
    
    // Property: For each query position t, at least one weight for positions <= t should be non-zero
    // (unless all Q*K products happen to be exactly zero, which is extremely unlikely)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                bool has_nonzero = false;
                for (int k_pos = 0; k_pos <= q_pos; ++k_pos) {
                    int idx = ((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos;
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
    // Constrain dimensions
    int batch_size = 1 + (std::abs(batch_raw) % 16);
    int seq_len = 4 + (std::abs(seq_raw) % 124);  // 4 to 128
    
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
            RC_ASSERT(val >= 0.0f);  // Softmax outputs should be non-negative
            sum += val;
        }
        // Sum should be close to 1.0 (within FP16 tolerance)
        RC_ASSERT(sum > 0.98f && sum < 1.02f);
    }
}

RC_GTEST_FIXTURE_PROP(AttentionPropertyTest, SoftmaxPreservesOrder,
                      (int batch_raw, int seq_raw, unsigned seed)) {
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
