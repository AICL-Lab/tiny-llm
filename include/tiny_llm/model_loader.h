#pragma once

#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/result.h"
#include "tiny_llm/types.h"
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

// Model loader class
class ModelLoader {
  public:
    // Load model from GGUF format
    static Result<ModelWeights> loadGGUF(const std::string &path, ModelConfig &config);

    // Load model from simple binary format
    static Result<ModelWeights> loadBin(const std::string &path, const ModelConfig &config);

    // Free model weights from GPU
    static void freeWeights(ModelWeights &weights);

  private:
    // Binary format helpers
    static Result<QuantizedWeight> loadQuantizedTensor(std::ifstream &file, int rows, int cols,
                                                       int group_size);

    // GPU transfer
    static Result<void> transferToDevice(ModelWeights &weights, const ModelConfig &config);
    static Result<QuantizedWeight> allocateAndTransferQuantized(const int8_t *host_data,
                                                                const half *host_scales, int rows,
                                                                int cols, int group_size);
};

// Simple binary format for testing
// Header: magic (4 bytes) + version (4 bytes) + config (ModelConfig)
// Then: embedding, layers, final_norm, lm_head
struct BinHeader {
    uint32_t    magic; // "TLLM"
    uint32_t    version;
    ModelConfig config;
};

constexpr uint32_t BIN_MAGIC = 0x4D4C4C54; // "TLLM"
constexpr uint32_t BIN_VERSION = 1;

} // namespace tiny_llm
