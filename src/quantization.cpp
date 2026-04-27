#include "tiny_llm/quantization.h"
#include "tiny_llm/logger.h"

#include <cmath>
#include <cstring>

namespace tiny_llm {

Result<std::vector<half>> convertF32ToF16(const float *f32_data, size_t num_elements) {
    if (f32_data == nullptr) {
        return Result<std::vector<half>>::err("convertF32ToF16: null pointer");
    }

    std::vector<half> f16_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        f16_data[i] = __float2half(f32_data[i]);
    }

    return Result<std::vector<half>>::ok(std::move(f16_data));
}

Result<std::vector<half>> convertF32ToF16(const std::vector<float> &f32_data) {
    return convertF32ToF16(f32_data.data(), f32_data.size());
}

Result<std::vector<half>> dequantizeQ4_0(const uint8_t *data, size_t num_blocks) {
    if (data == nullptr) {
        return Result<std::vector<half>>::err("dequantizeQ4_0: null pointer");
    }

    // Q4_0: 32 values per block, each block has 16 bytes (32 x 4-bit) + 2 bytes (half scale)
    // Total: 18 bytes per block -> 32 FP16 outputs
    constexpr size_t BLOCK_SIZE = 32;
    std::vector<half> result(num_blocks * BLOCK_SIZE);

    for (size_t b = 0; b < num_blocks; ++b) {
        // Each block: scale (half) + 16 bytes of packed 4-bit values
        const half  *scale = reinterpret_cast<const half *>(data + b * 18);
        const uint8_t *packed = data + b * 18 + 2;

        float scale_f = __half2float(*scale);

        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            // Each byte contains two 4-bit values
            uint8_t packed_byte = packed[i / 2];
            int8_t  value;
            if (i % 2 == 0) {
                // Lower 4 bits
                value = (packed_byte & 0x0F);
                // Convert from unsigned 4-bit to signed: 0-15 -> -8 to 7
                value = (value > 7) ? (value - 16) : value;
            } else {
                // Upper 4 bits
                value = (packed_byte >> 4);
                value = (value > 7) ? (value - 16) : value;
            }

            float dequantized = scale_f * static_cast<float>(value);
            result[b * BLOCK_SIZE + i] = __float2half(dequantized);
        }
    }

    return Result<std::vector<half>>::ok(std::move(result));
}

Result<std::vector<half>> dequantizeQ8_0(const uint8_t *data, size_t num_blocks) {
    if (data == nullptr) {
        return Result<std::vector<half>>::err("dequantizeQ8_0: null pointer");
    }

    // Q8_0: 32 values per block, each block has 32 bytes (32 x 8-bit) + 2 bytes (half scale)
    // Total: 34 bytes per block -> 32 FP16 outputs
    constexpr size_t BLOCK_SIZE = 32;
    std::vector<half> result(num_blocks * BLOCK_SIZE);

    for (size_t b = 0; b < num_blocks; ++b) {
        // Each block: scale (half) + 32 bytes of int8 values
        const half     *scale = reinterpret_cast<const half *>(data + b * 34);
        const int8_t *values = reinterpret_cast<const int8_t *>(data + b * 34 + 2);

        float scale_f = __half2float(*scale);

        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            float dequantized = scale_f * static_cast<float>(values[i]);
            result[b * BLOCK_SIZE + i] = __float2half(dequantized);
        }
    }

    return Result<std::vector<half>>::ok(std::move(result));
}

Result<std::pair<std::vector<int8_t>, std::vector<half>>>
quantizeF16ToW8A16(const half *f16_data, int rows, int cols, int group_size) {
    if (f16_data == nullptr) {
        return Result<std::pair<std::vector<int8_t>, std::vector<half>>>::err(
            "quantizeF16ToW8A16: null pointer");
    }

    if (rows <= 0 || cols <= 0 || group_size <= 0) {
        return Result<std::pair<std::vector<int8_t>, std::vector<half>>>::err(
            "quantizeF16ToW8A16: invalid dimensions");
    }

    size_t total_elements = static_cast<size_t>(rows) * cols;
    int    scale_rows = (rows + group_size - 1) / group_size;
    size_t scale_elements = static_cast<size_t>(scale_rows) * cols;

    std::vector<int8_t> quantized(total_elements);
    std::vector<half>   scales(scale_elements);

    // Process each column independently
    for (int c = 0; c < cols; ++c) {
        // Process groups within the column
        for (int g = 0; g < scale_rows; ++g) {
            int group_start = g * group_size;
            int group_end = std::min(group_start + group_size, rows);
            int actual_group_size = group_end - group_start;

            // Find max absolute value in group
            float max_abs = 0.0f;
            for (int r = group_start; r < group_end; ++r) {
                float val = __half2float(f16_data[r * cols + c]);
                max_abs = std::max(max_abs, std::abs(val));
            }

            // Calculate scale (avoid division by zero)
            float scale = max_abs / 127.0f;
            if (scale < 1e-10f) {
                scale = 1.0f; // Avoid division by zero for all-zero groups
            }

            scales[static_cast<size_t>(g) * cols + c] = __float2half(static_cast<half>(scale));

            // Quantize values
            for (int r = group_start; r < group_end; ++r) {
                float val = __half2float(f16_data[r * cols + c]);
                int   quantized_val = static_cast<int>(std::round(val / scale));
                // Clamp to int8 range
                quantized_val = std::max(-128, std::min(127, quantized_val));
                quantized[static_cast<size_t>(r) * cols + c] = static_cast<int8_t>(quantized_val);
            }
        }
    }

    return Result<std::pair<std::vector<int8_t>, std::vector<half>>>::ok(
        {std::move(quantized), std::move(scales)});
}

} // namespace tiny_llm
