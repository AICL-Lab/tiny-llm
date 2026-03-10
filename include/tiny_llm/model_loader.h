#pragma once

#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/result.h"
#include "tiny_llm/types.h"
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

// GGUF file format constants
constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian
constexpr uint32_t GGUF_VERSION_MIN = 2;
constexpr uint32_t GGUF_VERSION_MAX = 3;

// GGUF data types
enum class GGUFType : uint32_t {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12
};

// GGUF tensor types
enum class GGMLType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  I8 = 24,
  I16 = 25,
  I32 = 26
};

// GGUF header structure
struct GGUFHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

// GGUF tensor info
struct GGUFTensorInfo {
  std::string name;
  std::vector<uint64_t> dimensions;
  GGMLType type;
  uint64_t offset;
};

// Model loader class
class ModelLoader {
public:
  // Load model from GGUF format
  static Result<ModelWeights> loadGGUF(const std::string &path,
                                       ModelConfig &config);

  // Load model from simple binary format
  static Result<ModelWeights> loadBin(const std::string &path,
                                      const ModelConfig &config);

  // Free model weights from GPU
  static void freeWeights(ModelWeights &weights);

private:
  // GGUF parsing helpers
  static Result<GGUFHeader> parseGGUFHeader(std::ifstream &file);
  static Result<std::string> readString(std::ifstream &file);
  static Result<std::pair<std::string, std::string>>
  readMetadataKV(std::ifstream &file);
  static Result<GGUFTensorInfo> readTensorInfo(std::ifstream &file);

  // Binary format helpers
  static Result<QuantizedWeight>
  loadQuantizedTensor(std::ifstream &file, int rows, int cols, int group_size);

  // GPU transfer
  static Result<void> transferToDevice(ModelWeights &weights,
                                       const ModelConfig &config);
  static Result<QuantizedWeight>
  allocateAndTransferQuantized(const int8_t *host_data, const half *host_scales,
                               int rows, int cols, int group_size);
};

// Simple binary format for testing
// Header: magic (4 bytes) + version (4 bytes) + config (ModelConfig)
// Then: embedding, layers, final_norm, lm_head
struct BinHeader {
  uint32_t magic; // "TLLM"
  uint32_t version;
  ModelConfig config;
};

constexpr uint32_t BIN_MAGIC = 0x4D4C4C54; // "TLLM"
constexpr uint32_t BIN_VERSION = 1;

} // namespace tiny_llm
