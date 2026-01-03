#include "tiny_llm/model_loader.h"
#include <cstring>
#include <algorithm>

namespace tiny_llm {

Result<GGUFHeader> ModelLoader::parseGGUFHeader(std::ifstream& file) {
    GGUFHeader header;
    
    // Read magic number
    file.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
    if (!file) {
        return Result<GGUFHeader>::err("Failed to read GGUF magic number");
    }
    
    if (header.magic != GGUF_MAGIC) {
        return Result<GGUFHeader>::err(
            "Invalid GGUF magic number: expected 0x" + 
            std::to_string(GGUF_MAGIC) + ", got 0x" + std::to_string(header.magic));
    }
    
    // Read version
    file.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    if (!file) {
        return Result<GGUFHeader>::err("Failed to read GGUF version");
    }
    
    if (header.version < GGUF_VERSION_MIN || header.version > GGUF_VERSION_MAX) {
        return Result<GGUFHeader>::err(
            "Unsupported GGUF version: " + std::to_string(header.version) +
            " (supported: " + std::to_string(GGUF_VERSION_MIN) + "-" + 
            std::to_string(GGUF_VERSION_MAX) + ")");
    }
    
    // Read tensor count
    file.read(reinterpret_cast<char*>(&header.tensor_count), sizeof(header.tensor_count));
    if (!file) {
        return Result<GGUFHeader>::err("Failed to read tensor count");
    }
    
    // Read metadata KV count
    file.read(reinterpret_cast<char*>(&header.metadata_kv_count), sizeof(header.metadata_kv_count));
    if (!file) {
        return Result<GGUFHeader>::err("Failed to read metadata KV count");
    }
    
    return Result<GGUFHeader>::ok(header);
}

Result<std::string> ModelLoader::readString(std::ifstream& file) {
    uint64_t length;
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    if (!file) {
        return Result<std::string>::err("Failed to read string length");
    }
    
    if (length > 1024 * 1024) {  // Sanity check: 1MB max string
        return Result<std::string>::err("String too long: " + std::to_string(length));
    }
    
    std::string str(length, '\0');
    file.read(&str[0], length);
    if (!file) {
        return Result<std::string>::err("Failed to read string data");
    }
    
    return Result<std::string>::ok(str);
}

Result<GGUFTensorInfo> ModelLoader::readTensorInfo(std::ifstream& file) {
    GGUFTensorInfo info;
    
    // Read tensor name
    auto name_result = readString(file);
    if (name_result.isErr()) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor name: " + name_result.error());
    }
    info.name = name_result.value();
    
    // Read number of dimensions
    uint32_t n_dims;
    file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    if (!file) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor dimensions count");
    }
    
    // Read dimensions
    info.dimensions.resize(n_dims);
    for (uint32_t i = 0; i < n_dims; ++i) {
        file.read(reinterpret_cast<char*>(&info.dimensions[i]), sizeof(uint64_t));
        if (!file) {
            return Result<GGUFTensorInfo>::err("Failed to read tensor dimension " + std::to_string(i));
        }
    }
    
    // Read tensor type
    uint32_t type;
    file.read(reinterpret_cast<char*>(&type), sizeof(type));
    if (!file) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor type");
    }
    info.type = static_cast<GGMLType>(type);
    
    // Read offset
    file.read(reinterpret_cast<char*>(&info.offset), sizeof(info.offset));
    if (!file) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor offset");
    }
    
    return Result<GGUFTensorInfo>::ok(info);
}

Result<ModelWeights> ModelLoader::loadGGUF(const std::string& path, ModelConfig& config) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return Result<ModelWeights>::err("Failed to open file: " + path);
    }
    
    // Parse header
    auto header_result = parseGGUFHeader(file);
    if (header_result.isErr()) {
        return Result<ModelWeights>::err(header_result.error());
    }
    auto header = header_result.value();
    
    // Skip metadata for now (simplified implementation)
    // In a full implementation, we would parse metadata to get model config
    
    // For now, return error indicating GGUF support is partial
    return Result<ModelWeights>::err(
        "GGUF loading is partially implemented. Use loadBin() for testing. "
        "Header parsed successfully: version=" + std::to_string(header.version) +
        ", tensors=" + std::to_string(header.tensor_count));
}

Result<ModelWeights> ModelLoader::loadBin(const std::string& path, const ModelConfig& config) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return Result<ModelWeights>::err("Failed to open file: " + path);
    }
    
    // Read and validate header
    BinHeader header;
    file.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
    if (!file) {
        return Result<ModelWeights>::err("Failed to read binary header magic");
    }
    
    if (header.magic != BIN_MAGIC) {
        return Result<ModelWeights>::err(
            "Invalid binary magic number: expected 0x" + 
            std::to_string(BIN_MAGIC) + ", got 0x" + std::to_string(header.magic));
    }
    
    file.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    if (!file) {
        return Result<ModelWeights>::err("Failed to read binary version");
    }
    
    if (header.version != BIN_VERSION) {
        return Result<ModelWeights>::err(
            "Unsupported binary version: " + std::to_string(header.version));
    }
    
    // Read stored config
    file.read(reinterpret_cast<char*>(&header.config), sizeof(header.config));
    if (!file) {
        return Result<ModelWeights>::err("Failed to read model config");
    }
    
    // Validate dimensions match
    if (header.config.hidden_dim != config.hidden_dim ||
        header.config.num_layers != config.num_layers ||
        header.config.vocab_size != config.vocab_size) {
        return Result<ModelWeights>::err(
            "Model config mismatch: expected hidden_dim=" + std::to_string(config.hidden_dim) +
            ", num_layers=" + std::to_string(config.num_layers) +
            ", vocab_size=" + std::to_string(config.vocab_size) +
            ", got hidden_dim=" + std::to_string(header.config.hidden_dim) +
            ", num_layers=" + std::to_string(header.config.num_layers) +
            ", vocab_size=" + std::to_string(header.config.vocab_size));
    }
    
    ModelWeights weights;
    
    // Read token embedding [vocab_size, hidden_dim] as FP16
    size_t embed_size = static_cast<size_t>(config.vocab_size) * config.hidden_dim;
    std::vector<half> embed_host(embed_size);
    file.read(reinterpret_cast<char*>(embed_host.data()), embed_size * sizeof(half));
    if (!file) {
        return Result<ModelWeights>::err("Failed to read token embedding");
    }
    
    // Allocate and transfer embedding to GPU
    CUDA_CHECK(cudaMalloc(&weights.token_embedding, embed_size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(weights.token_embedding, embed_host.data(), 
                          embed_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Read layer weights
    weights.layers.resize(config.num_layers);
    int group_size = 128;  // Default group size
    
    for (int layer = 0; layer < config.num_layers; ++layer) {
        auto& lw = weights.layers[layer];
        
        // Read attention weights
        auto load_qweight = [&](QuantizedWeight& qw, int rows, int cols) -> Result<void> {
            auto result = loadQuantizedTensor(file, rows, cols, group_size);
            if (result.isErr()) {
                return Result<void>::err(result.error());
            }
            qw = result.value();
            return Result<void>::ok();
        };
        
        int hidden = config.hidden_dim;
        int kv_dim = config.num_kv_heads * config.head_dim;
        int inter = config.intermediate_dim;
        
        // Q, K, V, O projections
        auto r = load_qweight(lw.wq, hidden, hidden);
        if (r.isErr()) return Result<ModelWeights>::err("Layer " + std::to_string(layer) + " wq: " + r.error());
        
        r = load_qweight(lw.wk, hidden, kv_dim);
        if (r.isErr()) return Result<ModelWeights>::err("Layer " + std::to_string(layer) + " wk: " + r.error());
        
        r = load_qweight(lw.wv, hidden, kv_dim);
        if (r.isErr()) return Result<ModelWeights>::err("Layer " + std::to_string(layer) + " wv: " + r.error());
        
        r = load_qweight(lw.wo, hidden, hidden);
        if (r.isErr()) return Result<ModelWeights>::err("Layer " + std::to_string(layer) + " wo: " + r.error());
        
        // FFN weights
        r = load_qweight(lw.w1, hidden, inter);
        if (r.isErr()) return Result<ModelWeights>::err("Layer " + std::to_string(layer) + " w1: " + r.error());
        
        r = load_qweight(lw.w2, inter, hidden);
        if (r.isErr()) return Result<ModelWeights>::err("Layer " + std::to_string(layer) + " w2: " + r.error());
        
        r = load_qweight(lw.w3, hidden, inter);
        if (r.isErr()) return Result<ModelWeights>::err("Layer " + std::to_string(layer) + " w3: " + r.error());
        
        // RMSNorm weights (FP16)
        std::vector<half> norm_host(hidden);
        
        file.read(reinterpret_cast<char*>(norm_host.data()), hidden * sizeof(half));
        if (!file) return Result<ModelWeights>::err("Failed to read rms_att_weight for layer " + std::to_string(layer));
        CUDA_CHECK(cudaMalloc(&lw.rms_att_weight, hidden * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(lw.rms_att_weight, norm_host.data(), hidden * sizeof(half), cudaMemcpyHostToDevice));
        
        file.read(reinterpret_cast<char*>(norm_host.data()), hidden * sizeof(half));
        if (!file) return Result<ModelWeights>::err("Failed to read rms_ffn_weight for layer " + std::to_string(layer));
        CUDA_CHECK(cudaMalloc(&lw.rms_ffn_weight, hidden * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(lw.rms_ffn_weight, norm_host.data(), hidden * sizeof(half), cudaMemcpyHostToDevice));
    }
    
    // Read final norm weight
    std::vector<half> final_norm_host(config.hidden_dim);
    file.read(reinterpret_cast<char*>(final_norm_host.data()), config.hidden_dim * sizeof(half));
    if (!file) {
        freeWeights(weights);
        return Result<ModelWeights>::err("Failed to read final norm weight");
    }
    CUDA_CHECK(cudaMalloc(&weights.final_norm_weight, config.hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(weights.final_norm_weight, final_norm_host.data(), 
                          config.hidden_dim * sizeof(half), cudaMemcpyHostToDevice));
    
    // Read LM head
    auto lm_result = loadQuantizedTensor(file, config.hidden_dim, config.vocab_size, group_size);
    if (lm_result.isErr()) {
        freeWeights(weights);
        return Result<ModelWeights>::err("Failed to read LM head: " + lm_result.error());
    }
    weights.lm_head = lm_result.value();
    
    return Result<ModelWeights>::ok(std::move(weights));
}

Result<QuantizedWeight> ModelLoader::loadQuantizedTensor(
    std::ifstream& file, int rows, int cols, int group_size) {
    
    QuantizedWeight qw;
    qw.rows = rows;
    qw.cols = cols;
    qw.group_size = group_size;
    
    // Read INT8 weights
    size_t weight_size = qw.weightElements();
    std::vector<int8_t> weight_host(weight_size);
    file.read(reinterpret_cast<char*>(weight_host.data()), weight_size);
    if (!file) {
        return Result<QuantizedWeight>::err("Failed to read quantized weights");
    }
    
    // Read scales
    size_t scale_size = qw.scaleElements();
    std::vector<half> scale_host(scale_size);
    file.read(reinterpret_cast<char*>(scale_host.data()), scale_size * sizeof(half));
    if (!file) {
        return Result<QuantizedWeight>::err("Failed to read scale factors");
    }
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&qw.data, weight_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&qw.scales, scale_size * sizeof(half)));
    
    // Transfer to GPU
    CUDA_CHECK(cudaMemcpy(qw.data, weight_host.data(), weight_size * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(qw.scales, scale_host.data(), scale_size * sizeof(half), cudaMemcpyHostToDevice));
    
    return Result<QuantizedWeight>::ok(qw);
}

void ModelLoader::freeWeights(ModelWeights& weights) {
    if (weights.token_embedding) {
        cudaFree(weights.token_embedding);
        weights.token_embedding = nullptr;
    }
    
    for (auto& layer : weights.layers) {
        // Free quantized weights
        auto free_qw = [](QuantizedWeight& qw) {
            if (qw.data) { cudaFree(qw.data); qw.data = nullptr; }
            if (qw.scales) { cudaFree(qw.scales); qw.scales = nullptr; }
        };
        
        free_qw(layer.wq);
        free_qw(layer.wk);
        free_qw(layer.wv);
        free_qw(layer.wo);
        free_qw(layer.w1);
        free_qw(layer.w2);
        free_qw(layer.w3);
        
        if (layer.rms_att_weight) { cudaFree(layer.rms_att_weight); layer.rms_att_weight = nullptr; }
        if (layer.rms_ffn_weight) { cudaFree(layer.rms_ffn_weight); layer.rms_ffn_weight = nullptr; }
    }
    weights.layers.clear();
    
    if (weights.final_norm_weight) {
        cudaFree(weights.final_norm_weight);
        weights.final_norm_weight = nullptr;
    }
    
    if (weights.lm_head.data) { cudaFree(weights.lm_head.data); weights.lm_head.data = nullptr; }
    if (weights.lm_head.scales) { cudaFree(weights.lm_head.scales); weights.lm_head.scales = nullptr; }
}

} // namespace tiny_llm
