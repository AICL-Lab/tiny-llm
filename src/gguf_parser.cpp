#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/logger.h"

#include <algorithm>
#include <cstring>

namespace tiny_llm {

GGUFParser::GGUFParser(const std::string &path) : path_(path) {}

Result<void> GGUFParser::parse() {
    TLLM_INFO("Parsing GGUF file: {}", path_);

    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        return Result<void>::err("Failed to open file: " + path_);
    }

    // Parse header
    auto header_result = parseHeader(file);
    if (header_result.isErr()) {
        return header_result;
    }

    TLLM_INFO("GGUF header: version={}, tensors={}, metadata_entries={}", header_.version,
              header_.tensor_count, header_.metadata_kv_count);

    // Parse metadata
    auto meta_result = parseMetadata(file);
    if (meta_result.isErr()) {
        return meta_result;
    }

    TLLM_DEBUG("Parsed {} metadata entries", metadata_.kv.size());

    // Parse tensor info
    auto tensor_result = parseTensorInfo(file);
    if (tensor_result.isErr()) {
        return tensor_result;
    }

    TLLM_DEBUG("Parsed {} tensor entries", tensors_.size());

    // Calculate data offset with alignment
    data_offset_ = alignOffset(static_cast<uint64_t>(file.tellg()));

    TLLM_INFO("GGUF parsing complete. Data offset: {}", data_offset_);

    return Result<void>::ok();
}

Result<void> GGUFParser::parseHeader(std::ifstream &file) {
    file.read(reinterpret_cast<char *>(&header_.magic), 4);
    file.read(reinterpret_cast<char *>(&header_.version), 4);
    file.read(reinterpret_cast<char *>(&header_.tensor_count), 8);
    file.read(reinterpret_cast<char *>(&header_.metadata_kv_count), 8);

    if (!file) {
        return Result<void>::err("Failed to read GGUF header");
    }

    if (header_.magic != GGUF_MAGIC) {
        return Result<void>::err("Invalid GGUF magic number. Expected 0x" +
                                 std::to_string(GGUF_MAGIC) + ", got 0x" +
                                 std::to_string(header_.magic));
    }

    // Support GGUF version 2 and 3
    if (header_.version < 2 || header_.version > 3) {
        TLLM_WARN("GGUF version {} may not be fully supported", header_.version);
    }

    return Result<void>::ok();
}

Result<void> GGUFParser::parseMetadata(std::ifstream &file) {
    for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
        // Read key
        auto key_result = readString(file);
        if (key_result.isErr()) {
            return Result<void>::err("Failed to read metadata key: " + key_result.error());
        }
        std::string key = key_result.value();

        // Read value type
        uint32_t type_val;
        file.read(reinterpret_cast<char *>(&type_val), 4);
        if (!file) {
            return Result<void>::err("Failed to read metadata type for key: " + key);
        }

        // Read value
        auto value_result = readValue(file, static_cast<GGUFType>(type_val));
        if (value_result.isErr()) {
            return Result<void>::err("Failed to read metadata value for key: " + key + ": " +
                                     value_result.error());
        }

        metadata_.kv[key] = value_result.value();
        TLLM_TRACE("Metadata: {} = <value>", key);
    }

    return Result<void>::ok();
}

Result<void> GGUFParser::parseTensorInfo(std::ifstream &file) {
    tensors_.reserve(header_.tensor_count);

    for (uint64_t i = 0; i < header_.tensor_count; ++i) {
        auto tensor_result = readTensorInfoEntry(file);
        if (tensor_result.isErr()) {
            return Result<void>::err("Failed to read tensor info " + std::to_string(i) + ": " +
                                     tensor_result.error());
        }

        tensors_.push_back(tensor_result.value());
        tensor_name_map_[tensors_.back().name] = tensors_.size() - 1;

        TLLM_TRACE("Tensor: {} dims={} type={} offset={}", tensors_.back().name,
                   tensors_.back().dimensions.size(), static_cast<uint32_t>(tensors_.back().type),
                   tensors_.back().offset);
    }

    return Result<void>::ok();
}

Result<GGUFTensorInfo> GGUFParser::readTensorInfoEntry(std::ifstream &file) {
    GGUFTensorInfo info;

    // Read name
    auto name_result = readString(file);
    if (name_result.isErr()) {
        return Result<GGUFTensorInfo>::err(name_result.error());
    }
    info.name = name_result.value();

    // Read number of dimensions
    uint32_t n_dims;
    file.read(reinterpret_cast<char *>(&n_dims), 4);
    if (!file) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor n_dims");
    }

    // Read dimensions
    info.dimensions.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
        file.read(reinterpret_cast<char *>(&info.dimensions[d]), 8);
        if (!file) {
            return Result<GGUFTensorInfo>::err("Failed to read tensor dimension " +
                                               std::to_string(d));
        }
    }

    // Read type
    uint32_t type_val;
    file.read(reinterpret_cast<char *>(&type_val), 4);
    if (!file) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor type");
    }
    info.type = static_cast<GGMLType>(type_val);

    // Read offset
    file.read(reinterpret_cast<char *>(&info.offset), 8);
    if (!file) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor offset");
    }

    return Result<GGUFTensorInfo>::ok(info);
}

Result<GGUFValue> GGUFParser::readValue(std::ifstream &file, GGUFType type) {
    switch (type) {
    case GGUFType::UINT8: {
        uint8_t v;
        file.read(reinterpret_cast<char *>(&v), 1);
        if (!file) return Result<GGUFValue>::err("Failed to read uint8_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::INT8: {
        int8_t v;
        file.read(reinterpret_cast<char *>(&v), 1);
        if (!file) return Result<GGUFValue>::err("Failed to read int8_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::UINT16: {
        uint16_t v;
        file.read(reinterpret_cast<char *>(&v), 2);
        if (!file) return Result<GGUFValue>::err("Failed to read uint16_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::INT16: {
        int16_t v;
        file.read(reinterpret_cast<char *>(&v), 2);
        if (!file) return Result<GGUFValue>::err("Failed to read int16_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::UINT32: {
        uint32_t v;
        file.read(reinterpret_cast<char *>(&v), 4);
        if (!file) return Result<GGUFValue>::err("Failed to read uint32_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::INT32: {
        int32_t v;
        file.read(reinterpret_cast<char *>(&v), 4);
        if (!file) return Result<GGUFValue>::err("Failed to read int32_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::UINT64: {
        uint64_t v;
        file.read(reinterpret_cast<char *>(&v), 8);
        if (!file) return Result<GGUFValue>::err("Failed to read uint64_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::INT64: {
        int64_t v;
        file.read(reinterpret_cast<char *>(&v), 8);
        if (!file) return Result<GGUFValue>::err("Failed to read int64_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::FLOAT32: {
        float v;
        file.read(reinterpret_cast<char *>(&v), 4);
        if (!file) return Result<GGUFValue>::err("Failed to read float");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::FLOAT64: {
        double v;
        file.read(reinterpret_cast<char *>(&v), 8);
        if (!file) return Result<GGUFValue>::err("Failed to read double");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::BOOL: {
        uint8_t v;
        file.read(reinterpret_cast<char *>(&v), 1);
        if (!file) return Result<GGUFValue>::err("Failed to read bool");
        return Result<GGUFValue>::ok(GGUFValue{static_cast<bool>(v)});
    }
    case GGUFType::STRING: {
        auto str_result = readString(file);
        if (str_result.isErr()) {
            return Result<GGUFValue>::err(str_result.error());
        }
        return Result<GGUFValue>::ok(GGUFValue{str_result.value()});
    }
    case GGUFType::ARRAY: {
        auto arr_result = readArray(file);
        if (arr_result.isErr()) {
            return Result<GGUFValue>::err(arr_result.error());
        }
        return Result<GGUFValue>::ok(GGUFValue{arr_result.value()});
    }
    default:
        return Result<GGUFValue>::err("Unsupported GGUF type: " +
                                      std::to_string(static_cast<uint32_t>(type)));
    }
}

Result<GGUFValue> GGUFParser::readArray(std::ifstream &file) {
    // Read element type
    uint32_t type_val;
    file.read(reinterpret_cast<char *>(&type_val), 4);
    if (!file) {
        return Result<GGUFValue>::err("Failed to read array element type");
    }
    GGUFType elem_type = static_cast<GGUFType>(type_val);

    // Read count
    uint64_t count;
    file.read(reinterpret_cast<char *>(&count), 8);
    if (!file) {
        return Result<GGUFValue>::err("Failed to read array count");
    }

    TLLM_TRACE("Reading array of {} elements, type {}", count, static_cast<uint32_t>(elem_type));

    // Handle different array types
    switch (elem_type) {
    case GGUFType::UINT32: {
        std::vector<uint32_t> arr(count);
        file.read(reinterpret_cast<char *>(arr.data()), count * 4);
        if (!file) return Result<GGUFValue>::err("Failed to read uint32 array");
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    case GGUFType::INT32: {
        std::vector<int32_t> arr(count);
        file.read(reinterpret_cast<char *>(arr.data()), count * 4);
        if (!file) return Result<GGUFValue>::err("Failed to read int32 array");
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    case GGUFType::FLOAT32: {
        std::vector<float> arr(count);
        file.read(reinterpret_cast<char *>(arr.data()), count * 4);
        if (!file) return Result<GGUFValue>::err("Failed to read float array");
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    case GGUFType::FLOAT64: {
        std::vector<double> arr(count);
        file.read(reinterpret_cast<char *>(arr.data()), count * 8);
        if (!file) return Result<GGUFValue>::err("Failed to read double array");
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    case GGUFType::STRING: {
        std::vector<std::string> arr;
        arr.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            auto r = readString(file);
            if (r.isErr()) {
                return Result<GGUFValue>::err(r.error());
            }
            arr.push_back(r.value());
        }
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    default:
        // Skip unsupported array types
        TLLM_WARN("Unsupported array element type: {}, skipping {} elements",
                  static_cast<uint32_t>(elem_type), count);
        // Skip the data
        for (uint64_t i = 0; i < count; ++i) {
            auto r = readValue(file, elem_type);
            if (r.isErr()) {
                return Result<GGUFValue>::err(r.error());
            }
        }
        return Result<GGUFValue>::ok(GGUFValue{std::vector<uint8_t>()});
    }
}

Result<std::string> GGUFParser::readString(std::ifstream &file) {
    uint64_t length;
    file.read(reinterpret_cast<char *>(&length), 8);

    if (!file) {
        return Result<std::string>::err("Failed to read string length");
    }

    // Sanity check
    if (length > 1024 * 1024) {
        return Result<std::string>::err("String too long: " + std::to_string(length));
    }

    std::string str(length, '\0');
    file.read(&str[0], length);

    if (!file) {
        return Result<std::string>::err("Failed to read string data");
    }

    return Result<std::string>::ok(str);
}

uint64_t GGUFParser::alignOffset(uint64_t offset) const {
    return (offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

Result<ModelConfig> GGUFParser::extractModelConfig() const {
    ModelConfig config;

    // Helper lambdas for getting metadata values
    auto get_int = [this](const std::string &key, int &out) {
        auto it = metadata_.kv.find(key);
        if (it != metadata_.kv.end()) {
            if (auto *val = std::get_if<int32_t>(&it->second)) {
                out = *val;
                return true;
            }
            if (auto *val = std::get_if<uint32_t>(&it->second)) {
                out = static_cast<int>(*val);
                return true;
            }
            if (auto *val = std::get_if<int64_t>(&it->second)) {
                out = static_cast<int>(*val);
                return true;
            }
            if (auto *val = std::get_if<uint64_t>(&it->second)) {
                out = static_cast<int>(*val);
                return true;
            }
        }
        return false;
    };

    auto get_float = [this](const std::string &key, float &out) {
        auto it = metadata_.kv.find(key);
        if (it != metadata_.kv.end()) {
            if (auto *val = std::get_if<float>(&it->second)) {
                out = *val;
                return true;
            }
            if (auto *val = std::get_if<double>(&it->second)) {
                out = static_cast<float>(*val);
                return true;
            }
        }
        return false;
    };

    // Extract LLaMA-style metadata
    // Standard GGUF metadata keys
    get_int("llama.embedding_length", config.hidden_dim);
    get_int("llama.block_count", config.num_layers);
    get_int("llama.attention.head_count", config.num_heads);
    get_int("llama.attention.head_count_kv", config.num_kv_heads);
    get_int("llama.context_length", config.max_seq_len);
    get_int("general.architecture", config.hidden_dim); // fallback

    // Try alternative keys (some models use different naming)
    get_int("llama.embedding_length", config.hidden_dim);
    if (config.hidden_dim == 4096) { // default
        get_int("llama.embedding_length", config.hidden_dim);
    }

    // Tokenizer metadata
    get_int("tokenizer.ggml.model.vocab_size", config.vocab_size);
    get_int("tokenizer.ggml.eos_token_id", config.eos_token_id);
    get_int("tokenizer.ggml.bos_token_id", config.bos_token_id);

    // RoPE and normalization
    get_float("llama.attention.layer_norm_rms_epsilon", config.rms_norm_eps);
    get_float("llama.rope.freq_base", config.rope_theta);

    // FFN dimension
    get_int("llama.feed_forward_length", config.intermediate_dim);

    // Calculate derived values
    if (config.num_heads > 0) {
        config.head_dim = config.hidden_dim / config.num_heads;
    }

    // Validate essential fields
    if (config.hidden_dim <= 0 || config.num_layers <= 0 || config.num_heads <= 0 ||
        config.vocab_size <= 0) {
        TLLM_WARN("GGUF metadata may be incomplete. Using defaults for missing fields.");
        // Set reasonable defaults if missing
        if (config.hidden_dim <= 0) config.hidden_dim = 4096;
        if (config.num_layers <= 0) config.num_layers = 32;
        if (config.num_heads <= 0) config.num_heads = 32;
        if (config.num_kv_heads <= 0) config.num_kv_heads = config.num_heads;
        if (config.vocab_size <= 0) config.vocab_size = 32000;
        if (config.head_dim <= 0) config.head_dim = config.hidden_dim / config.num_heads;
        if (config.intermediate_dim <= 0) config.intermediate_dim = config.hidden_dim * 8 / 3;
    }

    TLLM_INFO("Extracted model config: hidden_dim={}, num_layers={}, num_heads={}, "
              "num_kv_heads={}, vocab_size={}",
              config.hidden_dim, config.num_layers, config.num_heads, config.num_kv_heads,
              config.vocab_size);

    return Result<ModelConfig>::ok(config);
}

const GGUFTensorInfo *GGUFParser::getTensorByName(const std::string &name) const {
    auto it = tensor_name_map_.find(name);
    if (it != tensor_name_map_.end()) {
        return &tensors_[it->second];
    }
    return nullptr;
}

Result<std::vector<uint8_t>> GGUFParser::readTensorData(const GGUFTensorInfo &tensor) {
    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        return Result<std::vector<uint8_t>>::err("Failed to open file: " + path_);
    }

    uint64_t read_offset = data_offset_ + tensor.offset;
    file.seekg(read_offset);

    if (!file) {
        return Result<std::vector<uint8_t>>::err("Failed to seek to tensor data at offset " +
                                                 std::to_string(read_offset));
    }

    size_t               size = tensor.calculateSize();
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char *>(data.data()), size);

    if (!file) {
        return Result<std::vector<uint8_t>>::err("Failed to read tensor data");
    }

    return Result<std::vector<uint8_t>>::ok(data);
}

// GGUFMetadata methods
bool GGUFMetadata::has(const std::string &key) const { return kv.find(key) != kv.end(); }

// GGUFTensorInfo methods
size_t GGUFTensorInfo::numElements() const {
    if (dimensions.empty()) return 0;
    size_t n = 1;
    for (auto d : dimensions) {
        n *= d;
    }
    return n;
}

size_t GGUFTensorInfo::calculateSize() const {
    size_t num_elem = numElements();

    // Bytes per element for each type
    switch (type) {
    case GGMLType::F32:
        return num_elem * 4;
    case GGMLType::F16:
        return num_elem * 2;
    case GGMLType::I8:
        return num_elem;
    case GGMLType::I16:
        return num_elem * 2;
    case GGMLType::I32:
        return num_elem * 4;
    case GGMLType::I64:
        return num_elem * 8;
    case GGMLType::F64:
        return num_elem * 8;
    case GGMLType::Q8_0:
        // Q8_0: 32 values per block, each block has 32 int8 + 1 half scale
        return (num_elem / 32) * (32 + 2);
    case GGMLType::Q4_0:
        // Q4_0: 32 values per block, each block has 16 int8 + 1 half scale
        return (num_elem / 32) * (16 + 2);
    case GGMLType::Q4_1:
        // Q4_1: 32 values per block, each block has 16 int8 + 2 half (scale + min)
        return (num_elem / 32) * (16 + 4);
    default:
        // Default to 2 bytes per element (FP16)
        TLLM_WARN("Unknown tensor type {}, assuming FP16 size", static_cast<uint32_t>(type));
        return num_elem * 2;
    }
}

} // namespace tiny_llm
