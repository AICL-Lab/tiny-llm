#include "tiny_llm/model_loader.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/logger.h"
#include "tiny_llm/quantization.h"

#include <algorithm>
#include <cstring>

namespace tiny_llm {

Result<ModelWeights> ModelLoader::loadGGUF(const std::string &path, ModelConfig &config) {
    TLLM_INFO("Loading GGUF model from: {}", path);

    // Use the new GGUFParser
    GGUFParser parser(path);
    auto       parse_result = parser.parse();
    if (parse_result.isErr()) {
        TLLM_ERROR("Failed to parse GGUF: {}", parse_result.error());
        return Result<ModelWeights>::err("Failed to parse GGUF: " + parse_result.error());
    }

    // Extract model config
    auto config_result = parser.extractModelConfig();
    if (config_result.isErr()) {
        TLLM_ERROR("Failed to extract model config: {}", config_result.error());
        return Result<ModelWeights>::err("Failed to extract model config: " +
                                         config_result.error());
    }
    config = config_result.value();

    TLLM_INFO("Model config: hidden_dim={}, num_layers={}, vocab_size={}", config.hidden_dim,
              config.num_layers, config.vocab_size);

    const auto &tensors = parser.getTensors();
    TLLM_DEBUG("Found {} tensors", tensors.size());

    // Build tensor name map for quick lookup
    std::unordered_map<std::string, const GGUFTensorInfo *> tensor_map;
    for (const auto &t : tensors) {
        tensor_map[t.name] = &t;
    }

    // Helper function to find tensor
    auto find_tensor = [&](const std::string &name) -> const GGUFTensorInfo * {
        auto it = tensor_map.find(name);
        if (it != tensor_map.end()) {
            return it->second;
        }
        return nullptr;
    };

    auto find_any_tensor =
        [&](std::initializer_list<const char *> names) -> const GGUFTensorInfo * {
        for (const char *name : names) {
            if (const auto *tensor = find_tensor(name)) {
                return tensor;
            }
        }
        return nullptr;
    };

    std::vector<std::string> missing_tensors;
    auto                     require_tensor = [&](std::initializer_list<const char *> names) {
        if (!find_any_tensor(names)) {
            std::string joined;
            for (const char *name : names) {
                if (!joined.empty()) {
                    joined += " | ";
                }
                joined += name;
            }
            missing_tensors.push_back(std::move(joined));
        }
    };

    require_tensor({"token_embd.weight", "tok_embeddings.weight"});
    for (int layer = 0; layer < config.num_layers; ++layer) {
        const std::string layer_prefix = "blk." + std::to_string(layer) + ".";
        const std::string llama_prefix = "layers." + std::to_string(layer) + ".";
        auto              require_layer_tensor = [&](std::initializer_list<const char *> suffixes) {
            for (const char *suffix : suffixes) {
                if (find_tensor(layer_prefix + suffix) || find_tensor(llama_prefix + suffix)) {
                    return;
                }
            }
            std::string joined;
            for (const char *suffix : suffixes) {
                if (!joined.empty()) {
                    joined += " | ";
                }
                joined += layer_prefix + suffix;
                joined += " | ";
                joined += llama_prefix + suffix;
            }
            missing_tensors.push_back(std::move(joined));
        };

        require_layer_tensor({"attn_q.weight"});
        require_layer_tensor({"attn_k.weight"});
        require_layer_tensor({"attn_v.weight"});
        require_layer_tensor({"attn_output.weight", "attn_out.weight"});
        require_layer_tensor({"ffn_gate.weight"});
        require_layer_tensor({"ffn_down.weight"});
        require_layer_tensor({"ffn_up.weight"});
        require_layer_tensor({"attn_norm.weight"});
        require_layer_tensor({"ffn_norm.weight"});
    }

    require_tensor({"output_norm.weight", "norm.weight"});
    require_tensor({"output.weight", "lm_head.weight"});

    if (!missing_tensors.empty()) {
        std::string missing;
        for (size_t i = 0; i < missing_tensors.size(); ++i) {
            if (i > 0) {
                missing += ", ";
            }
            missing += missing_tensors[i];
        }
        return Result<ModelWeights>::err("GGUF runtime tensors missing: " + missing);
    }

    return Result<ModelWeights>::err(
        "GGUF runtime tensor conversion is not implemented; use GGUF parsing/metadata surfaces "
        "only or convert to the supported binary runtime format first.");
}

Result<ModelWeights> ModelLoader::loadBin(const std::string &path, const ModelConfig &config) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return Result<ModelWeights>::err("Failed to open file: " + path);
    }

    // Read and validate header
    BinHeader header;
    file.read(reinterpret_cast<char *>(&header.magic), sizeof(header.magic));
    if (!file) {
        return Result<ModelWeights>::err("Failed to read binary header magic");
    }

    if (header.magic != BIN_MAGIC) {
        return Result<ModelWeights>::err("Invalid binary magic number: expected 0x" +
                                         std::to_string(BIN_MAGIC) + ", got 0x" +
                                         std::to_string(header.magic));
    }

    file.read(reinterpret_cast<char *>(&header.version), sizeof(header.version));
    if (!file) {
        return Result<ModelWeights>::err("Failed to read binary version");
    }

    if (header.version != BIN_VERSION) {
        return Result<ModelWeights>::err("Unsupported binary version: " +
                                         std::to_string(header.version));
    }

    // Read stored config
    file.read(reinterpret_cast<char *>(&header.config), sizeof(header.config));
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
    bool         success = false;
    auto         cleanup_on_error = [&]() {
        if (!success) {
            freeWeights(weights);
        }
    };

    // Read token embedding [vocab_size, hidden_dim] as FP16
    size_t            embed_size = static_cast<size_t>(config.vocab_size) * config.hidden_dim;
    std::vector<half> embed_host(embed_size);
    file.read(reinterpret_cast<char *>(embed_host.data()), embed_size * sizeof(half));
    if (!file) {
        cleanup_on_error();
        return Result<ModelWeights>::err("Failed to read token embedding");
    }

    // Allocate and transfer embedding to GPU
    CUDA_CHECK(cudaMalloc(&weights.token_embedding, embed_size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(weights.token_embedding, embed_host.data(), embed_size * sizeof(half),
                          cudaMemcpyHostToDevice));

    // Read layer weights
    weights.layers.resize(config.num_layers);
    int group_size = 128; // Default group size

    for (int layer = 0; layer < config.num_layers; ++layer) {
        auto &lw = weights.layers[layer];

        // Read attention weights
        auto load_qweight = [&](QuantizedWeight &qw, int rows, int cols) -> Result<void> {
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
        if (r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                             " wq: " + r.error());
        }

        r = load_qweight(lw.wk, hidden, kv_dim);
        if (r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                             " wk: " + r.error());
        }

        r = load_qweight(lw.wv, hidden, kv_dim);
        if (r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                             " wv: " + r.error());
        }

        r = load_qweight(lw.wo, hidden, hidden);
        if (r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                             " wo: " + r.error());
        }

        // FFN weights
        r = load_qweight(lw.w1, hidden, inter);
        if (r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                             " w1: " + r.error());
        }

        r = load_qweight(lw.w2, inter, hidden);
        if (r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                             " w2: " + r.error());
        }

        r = load_qweight(lw.w3, hidden, inter);
        if (r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                             " w3: " + r.error());
        }

        // RMSNorm weights (FP16)
        std::vector<half> norm_host(hidden);

        file.read(reinterpret_cast<char *>(norm_host.data()), hidden * sizeof(half));
        if (!file) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Failed to read rms_att_weight for layer " +
                                             std::to_string(layer));
        }
        CUDA_CHECK(cudaMalloc(&lw.rms_att_weight, hidden * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(lw.rms_att_weight, norm_host.data(), hidden * sizeof(half),
                              cudaMemcpyHostToDevice));

        file.read(reinterpret_cast<char *>(norm_host.data()), hidden * sizeof(half));
        if (!file) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Failed to read rms_ffn_weight for layer " +
                                             std::to_string(layer));
        }
        CUDA_CHECK(cudaMalloc(&lw.rms_ffn_weight, hidden * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(lw.rms_ffn_weight, norm_host.data(), hidden * sizeof(half),
                              cudaMemcpyHostToDevice));
    }

    // Read final norm weight
    std::vector<half> final_norm_host(config.hidden_dim);
    file.read(reinterpret_cast<char *>(final_norm_host.data()), config.hidden_dim * sizeof(half));
    if (!file) {
        cleanup_on_error();
        return Result<ModelWeights>::err("Failed to read final norm weight");
    }
    CUDA_CHECK(cudaMalloc(&weights.final_norm_weight, config.hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(weights.final_norm_weight, final_norm_host.data(),
                          config.hidden_dim * sizeof(half), cudaMemcpyHostToDevice));

    // Read LM head
    auto lm_result = loadQuantizedTensor(file, config.hidden_dim, config.vocab_size, group_size);
    if (lm_result.isErr()) {
        cleanup_on_error();
        return Result<ModelWeights>::err("Failed to read LM head: " + lm_result.error());
    }
    weights.lm_head = lm_result.value();

    success = true;
    return Result<ModelWeights>::ok(std::move(weights));
}

Result<QuantizedWeight> ModelLoader::loadQuantizedTensor(std::ifstream &file, int rows, int cols,
                                                         int group_size) {

    QuantizedWeight qw;
    qw.rows = rows;
    qw.cols = cols;
    qw.group_size = group_size;

    // Read INT8 weights
    size_t              weight_size = qw.weightElements();
    std::vector<int8_t> weight_host(weight_size);
    file.read(reinterpret_cast<char *>(weight_host.data()), weight_size);
    if (!file) {
        return Result<QuantizedWeight>::err("Failed to read quantized weights");
    }

    // Read scales
    size_t            scale_size = qw.scaleElements();
    std::vector<half> scale_host(scale_size);
    file.read(reinterpret_cast<char *>(scale_host.data()), scale_size * sizeof(half));
    if (!file) {
        return Result<QuantizedWeight>::err("Failed to read scale factors");
    }

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&qw.data, weight_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&qw.scales, scale_size * sizeof(half)));

    // Transfer to GPU
    CUDA_CHECK(cudaMemcpy(qw.data, weight_host.data(), weight_size * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(qw.scales, scale_host.data(), scale_size * sizeof(half),
                          cudaMemcpyHostToDevice));

    return Result<QuantizedWeight>::ok(qw);
}

void ModelLoader::freeWeights(ModelWeights &weights) {
    if (weights.token_embedding) {
        cudaFree(weights.token_embedding);
        weights.token_embedding = nullptr;
    }

    for (auto &layer : weights.layers) {
        // Free quantized weights
        auto free_qw = [](QuantizedWeight &qw) {
            if (qw.data) {
                cudaFree(qw.data);
                qw.data = nullptr;
            }
            if (qw.scales) {
                cudaFree(qw.scales);
                qw.scales = nullptr;
            }
        };

        free_qw(layer.wq);
        free_qw(layer.wk);
        free_qw(layer.wv);
        free_qw(layer.wo);
        free_qw(layer.w1);
        free_qw(layer.w2);
        free_qw(layer.w3);

        if (layer.rms_att_weight) {
            cudaFree(layer.rms_att_weight);
            layer.rms_att_weight = nullptr;
        }
        if (layer.rms_ffn_weight) {
            cudaFree(layer.rms_ffn_weight);
            layer.rms_ffn_weight = nullptr;
        }
    }
    weights.layers.clear();

    if (weights.final_norm_weight) {
        cudaFree(weights.final_norm_weight);
        weights.final_norm_weight = nullptr;
    }

    if (weights.lm_head.data) {
        cudaFree(weights.lm_head.data);
        weights.lm_head.data = nullptr;
    }
    if (weights.lm_head.scales) {
        cudaFree(weights.lm_head.scales);
        weights.lm_head.scales = nullptr;
    }
}

} // namespace tiny_llm
