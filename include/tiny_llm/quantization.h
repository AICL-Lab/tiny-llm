#pragma once

#include "result.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <vector>

namespace tiny_llm {

/**
 * @brief Convert F32 data to F16
 * @param f32_data Pointer to F32 data
 * @param num_elements Number of elements to convert
 * @return Result containing vector of F16 data or error
 */
Result<std::vector<half>> convertF32ToF16(const float *f32_data, size_t num_elements);

/**
 * @brief Convert F32 vector to F16 vector
 * @param f32_data Vector of F32 data
 * @return Result containing vector of F16 data or error
 */
Result<std::vector<half>> convertF32ToF16(const std::vector<float> &f32_data);

/**
 * @brief Dequantize Q4_0 block to F16
 * Q4_0 format: 32 values per block, each block has 32 int4 values + 1 half scale
 * @param data Raw Q4_0 data
 * @param num_blocks Number of blocks
 * @return Result containing vector of F16 values or error
 */
Result<std::vector<half>> dequantizeQ4_0(const uint8_t *data, size_t num_blocks);

/**
 * @brief Dequantize Q8_0 block to F16
 * Q8_0 format: 32 values per block, each block has 32 int8 values + 1 half scale
 * @param data Raw Q8_0 data
 * @param num_blocks Number of blocks
 * @return Result containing vector of F16 values or error
 */
Result<std::vector<half>> dequantizeQ8_0(const uint8_t *data, size_t num_blocks);

/**
 * @brief Quantize F16 values to W8A16 format (int8 with per-group scales)
 * @param f16_data F16 input data
 * @param rows Number of rows
 * @param cols Number of columns
 * @param group_size Number of elements per scale group (default 128)
 * @return Result containing pair (int8_data, scales) or error
 */
Result<std::pair<std::vector<int8_t>, std::vector<half>>>
quantizeF16ToW8A16(const half *f16_data, int rows, int cols, int group_size = 128);

} // namespace tiny_llm
