#include "tiny_llm/cuda_utils.h"
#include "w8a16_matmul.cuh"
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
        int         device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        has_device = (err == cudaSuccess && device_count > 0);
        checked = true;
    }
    return has_device;
}

// Helper class for GPU test fixtures
class W8A16MatMulTest : public ::testing::Test {
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

    // Generate random FP16 matrix
    std::vector<half> randomFP16(int rows, int cols, float scale = 1.0f) {
        std::vector<half>                     data(rows * cols);
        std::mt19937                          gen(42);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto &v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }

    // Generate random INT8 weights
    std::vector<int8_t> randomINT8(int rows, int cols) {
        std::vector<int8_t>                data(rows * cols);
        std::mt19937                       gen(123);
        std::uniform_int_distribution<int> dist(-127, 127);
        for (auto &v : data) {
            v = static_cast<int8_t>(dist(gen));
        }
        return data;
    }

    // Generate random scales
    std::vector<half> randomScales(int num_groups, int cols) {
        std::vector<half>                     data(num_groups * cols);
        std::mt19937                          gen(456);
        std::uniform_real_distribution<float> dist(0.001f, 0.1f);
        for (auto &v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }

    // Compute relative error
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

// Unit tests
TEST_F(W8A16MatMulTest, SmallMatrixCorrectness) {
    int M = 4, N = 8, K = 16;
    int group_size = 8;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    // Allocate device memory
    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);
    DeviceBuffer<half>   d_weight_fp16(K * N);
    DeviceBuffer<half>   d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    // Run W8A16 kernel
    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                 group_size);

    // Dequantize and run FP16 reference
    dequantize_weights(d_weight.data(), d_scales.data(), d_weight_fp16.data(), K, N, group_size);
    fp16_matmul_reference(d_input.data(), d_weight_fp16.data(), d_output_ref.data(), M, N, K);

    cudaDeviceSynchronize();

    // Copy results back
    std::vector<half> output(M * N);
    std::vector<half> output_ref(M * N);
    d_output.copyToHost(output.data(), M * N);
    d_output_ref.copyToHost(output_ref.data(), M * N);
    cudaDeviceSynchronize();

    // Check relative error
    float rel_error = computeRelativeError(output, output_ref);
    EXPECT_LT(rel_error, 0.01f) << "Relative error too high: " << rel_error;
}

TEST_F(W8A16MatMulTest, EdgeCaseM1) {
    int M = 1, N = 64, K = 128;
    int group_size = 32;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    EXPECT_NO_THROW({
        w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                     group_size);
        cudaDeviceSynchronize();
    });
}

TEST_F(W8A16MatMulTest, EdgeCaseN1) {
    int M = 64, N = 1, K = 128;
    int group_size = 32;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    EXPECT_NO_THROW({
        w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                     group_size);
        cudaDeviceSynchronize();
    });
}

TEST_F(W8A16MatMulTest, NonAlignedDimensions) {
    int M = 17, N = 33, K = 65;
    int group_size = 32;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);
    DeviceBuffer<half>   d_weight_fp16(K * N);
    DeviceBuffer<half>   d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                 group_size);

    dequantize_weights(d_weight.data(), d_scales.data(), d_weight_fp16.data(), K, N, group_size);
    fp16_matmul_reference(d_input.data(), d_weight_fp16.data(), d_output_ref.data(), M, N, K);

    cudaDeviceSynchronize();

    std::vector<half> output(M * N);
    std::vector<half> output_ref(M * N);
    d_output.copyToHost(output.data(), M * N);
    d_output_ref.copyToHost(output_ref.data(), M * N);
    cudaDeviceSynchronize();

    float rel_error = computeRelativeError(output, output_ref);
    EXPECT_LT(rel_error, 0.01f);
}

// Property-based tests
// Feature: tiny-llm-inference-engine, Property 1: W8A16 MatMul Numerical
// Accuracy Validates: Requirements 2.5, 2.6
// NOTE: Property-based tests are disabled when no CUDA device is available

class W8A16PropertyTest : public W8A16MatMulTest {
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

RC_GTEST_FIXTURE_PROP(W8A16PropertyTest, NumericalAccuracyProperty,
                      (int m_raw, int n_raw, int k_raw)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions to reasonable ranges
    int M = 1 + (std::abs(m_raw) % 128);
    int N = 8 + (std::abs(n_raw) % 256);
    int K = 32 + (std::abs(k_raw) % 512);
    int group_size = 32;
    int num_groups = (K + group_size - 1) / group_size;

    // Generate random data
    auto input = randomFP16(M, K, 0.5f);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    // Allocate device memory
    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);
    DeviceBuffer<half>   d_weight_fp16(K * N);
    DeviceBuffer<half>   d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    // Run W8A16 kernel
    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                 group_size);

    // Dequantize and run FP16 reference
    dequantize_weights(d_weight.data(), d_scales.data(), d_weight_fp16.data(), K, N, group_size);
    fp16_matmul_reference(d_input.data(), d_weight_fp16.data(), d_output_ref.data(), M, N, K);

    cudaDeviceSynchronize();

    // Copy results back
    std::vector<half> output(M * N);
    std::vector<half> output_ref(M * N);
    d_output.copyToHost(output.data(), M * N);
    d_output_ref.copyToHost(output_ref.data(), M * N);
    cudaDeviceSynchronize();

    // Property: relative error should be < 1%
    float rel_error = computeRelativeError(output, output_ref);
    RC_ASSERT(rel_error < 0.01f);
}

RC_GTEST_FIXTURE_PROP(W8A16PropertyTest, DifferentGroupSizes,
                      (int m_raw, int n_raw, int k_raw, int gs_raw)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    int M = 1 + (std::abs(m_raw) % 64);
    int N = 8 + (std::abs(n_raw) % 128);
    int K = 64 + (std::abs(k_raw) % 256);

    // Test different group sizes: 32, 64, 128
    int group_sizes[] = {32, 64, 128};
    int group_size = group_sizes[std::abs(gs_raw) % 3];
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K, 0.5f);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);
    DeviceBuffer<half>   d_weight_fp16(K * N);
    DeviceBuffer<half>   d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                 group_size);

    dequantize_weights(d_weight.data(), d_scales.data(), d_weight_fp16.data(), K, N, group_size);
    fp16_matmul_reference(d_input.data(), d_weight_fp16.data(), d_output_ref.data(), M, N, K);

    cudaDeviceSynchronize();

    std::vector<half> output(M * N);
    std::vector<half> output_ref(M * N);
    d_output.copyToHost(output.data(), M * N);
    d_output_ref.copyToHost(output_ref.data(), M * N);
    cudaDeviceSynchronize();

    float rel_error = computeRelativeError(output, output_ref);
    RC_ASSERT(rel_error < 0.01f);
}
