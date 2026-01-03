#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "tiny_llm/types.h"

using namespace tiny_llm;

// Unit tests for QuantizedWeight
class QuantizedWeightTest : public ::testing::Test {};

TEST_F(QuantizedWeightTest, DefaultConstructorCreatesInvalidWeight) {
    QuantizedWeight weight;
    EXPECT_FALSE(weight.isValid());
}

TEST_F(QuantizedWeightTest, ScaleDimensionsCalculation) {
    QuantizedWeight weight;
    weight.rows = 256;
    weight.cols = 512;
    weight.group_size = 128;
    
    // Expected: scale cols = ceil(512 / 128) = 4
    EXPECT_EQ(weight.scaleRows(), 256);
    EXPECT_EQ(weight.scaleCols(), 4);
}

TEST_F(QuantizedWeightTest, ScaleDimensionsWithNonDivisible) {
    QuantizedWeight weight;
    weight.rows = 100;
    weight.cols = 300;
    weight.group_size = 128;
    
    // Expected: scale cols = ceil(300 / 128) = 3
    EXPECT_EQ(weight.scaleRows(), 100);
    EXPECT_EQ(weight.scaleCols(), 3);
}

TEST_F(QuantizedWeightTest, MemorySizeCalculation) {
    QuantizedWeight weight;
    weight.rows = 256;
    weight.cols = 512;
    weight.group_size = 128;
    
    EXPECT_EQ(weight.weightElements(), 256 * 512);
    EXPECT_EQ(weight.weightBytes(), 256 * 512 * sizeof(int8_t));
    
    // scale elements = 256 * 4 = 1024
    EXPECT_EQ(weight.scaleElements(), 256 * 4);
    EXPECT_EQ(weight.scaleBytes(), 256 * 4 * sizeof(half));
}

// Property-based tests for Weight-Scale Dimension Consistency
// Feature: tiny-llm-inference-engine, Property 8: Weight-Scale Dimension Consistency
// Validates: Requirements 1.3, 7.2

RC_GTEST_PROP(QuantizedWeightProperty, ScaleDimensionsAreConsistent,
              (int rows, int cols, int group_size)) {
    // Constrain inputs to reasonable ranges
    rows = 1 + (std::abs(rows) % 4096);
    cols = 1 + (std::abs(cols) % 4096);
    group_size = 1 + (std::abs(group_size) % 256);
    
    QuantizedWeight weight;
    weight.rows = rows;
    weight.cols = cols;
    weight.group_size = group_size;
    
    // Property: scale tensor must have shape [rows, ceil(cols / group_size)]
    int expected_scale_rows = rows;
    int expected_scale_cols = (cols + group_size - 1) / group_size;
    
    RC_ASSERT(weight.scaleRows() == expected_scale_rows);
    RC_ASSERT(weight.scaleCols() == expected_scale_cols);
    
    // Property: scale elements must cover all weight columns
    RC_ASSERT(weight.scaleCols() * group_size >= cols);
    
    // Property: scale elements should not be more than necessary
    RC_ASSERT((weight.scaleCols() - 1) * group_size < cols);
}

RC_GTEST_PROP(QuantizedWeightProperty, TotalBytesIsCorrect,
              (int rows, int cols, int group_size)) {
    rows = 1 + (std::abs(rows) % 2048);
    cols = 1 + (std::abs(cols) % 2048);
    group_size = 1 + (std::abs(group_size) % 256);
    
    QuantizedWeight weight;
    weight.rows = rows;
    weight.cols = cols;
    weight.group_size = group_size;
    
    // Property: total bytes = weight bytes + scale bytes
    size_t expected_total = weight.weightBytes() + weight.scaleBytes();
    RC_ASSERT(weight.totalBytes() == expected_total);
    
    // Property: weight bytes = rows * cols * sizeof(int8_t)
    RC_ASSERT(weight.weightBytes() == static_cast<size_t>(rows) * cols * sizeof(int8_t));
}

RC_GTEST_PROP(QuantizedWeightProperty, GroupSizeAffectsScaleSize,
              (int rows, int cols)) {
    rows = 64 + (std::abs(rows) % 1024);
    cols = 128 + (std::abs(cols) % 2048);
    
    QuantizedWeight weight1, weight2;
    weight1.rows = weight2.rows = rows;
    weight1.cols = weight2.cols = cols;
    weight1.group_size = 64;
    weight2.group_size = 128;
    
    // Property: larger group size means fewer scale elements
    RC_ASSERT(weight1.scaleElements() >= weight2.scaleElements());
    
    // Property: smaller group size means more memory for scales
    RC_ASSERT(weight1.scaleBytes() >= weight2.scaleBytes());
}

// Test ModelConfig defaults
class ModelConfigTest : public ::testing::Test {};

TEST_F(ModelConfigTest, DefaultValues) {
    ModelConfig config;
    EXPECT_EQ(config.vocab_size, 32000);
    EXPECT_EQ(config.hidden_dim, 4096);
    EXPECT_EQ(config.num_layers, 32);
    EXPECT_EQ(config.num_heads, 32);
    EXPECT_EQ(config.head_dim, 128);
    EXPECT_GT(config.rms_norm_eps, 0.0f);
}

// Test GenerationConfig
class GenerationConfigTest : public ::testing::Test {};

TEST_F(GenerationConfigTest, DefaultValues) {
    GenerationConfig config;
    EXPECT_EQ(config.max_new_tokens, 256);
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_EQ(config.top_k, 50);
    EXPECT_FALSE(config.do_sample);
}

// Test KVCacheConfig
class KVCacheConfigTest : public ::testing::Test {};

TEST_F(KVCacheConfigTest, DefaultValues) {
    KVCacheConfig config;
    EXPECT_EQ(config.num_layers, 32);
    EXPECT_EQ(config.num_heads, 32);
    EXPECT_EQ(config.head_dim, 128);
    EXPECT_EQ(config.max_seq_len, 2048);
}
