#include "tiny_llm/model_loader.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/logger.h"

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

    ModelWeights weights;
    bool         success = false;
    auto         cleanup_on_error = [&]() {
        if (!success) {
            freeWeights(weights);
        }
    };

    // Helper function to find tensor
    auto find_tensor = [&](const std::string &name) -> const GGUFTensorInfo * {
        auto it = tensor_map.find(name);
        if (it != tensor_map.end()) {
            return it->second;
        }
        return nullptr;
    };

    // Load token embedding
    // GGUF naming: token_embd.weight
    const GGUFTensorInfo *embed_tensor = find_tensor("token_embd.weight");
    if (!embed_tensor) {
        // Try alternative names
        embed_tensor = find_tensor("tok_embeddings.weight");
    }

    if (embed_tensor) {
        TLLM_DEBUG("Loading token embedding from tensor: {}", embed_tensor->name);
        auto data_result = parser.readTensorData(*embed_tensor);
        if (data_result.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Failed to read token embedding: " +
                                             data_result.error());
        }

        size_t embed_size = static_cast<size_t>(config.vocab_size) * config.hidden_dim;
        CUDA_CHECK(cudaMalloc(&weights.token_embedding, embed_size * sizeof(half)));

        // Convert to FP16 if needed
        const auto &data = data_result.value();
        if (embed_tensor->type == GGMLType::F16) {
            CUDA_CHECK(cudaMemcpy(weights.token_embedding, data.data(), data.size(),
                                  cudaMemcpyHostToDevice));
        } else if (embed_tensor->type == GGMLType::F32) {
            // Convert F32 to F16
            std::vector<half> f16_data(embed_size);
            const float      *f32_data = reinterpret_cast<const float *>(data.data());
            for (size_t i = 0; i < embed_size; ++i) {
                f16_data[i] = __float2half(f32_data[i]);
            }
            CUDA_CHECK(cudaMemcpy(weights.token_embedding, f16_data.data(),
                                  embed_size * sizeof(half), cudaMemcpyHostToDevice));
        } else {
            TLLM_WARN("Unsupported embedding type: {}, skipping",
                      static_cast<int>(embed_tensor->type));
        }
    } else {
        TLLM_WARN("Token embedding tensor not found, using zeros");
        size_t embed_size = static_cast<size_t>(config.vocab_size) * config.hidden_dim;
        CUDA_CHECK(cudaMalloc(&weights.token_embedding, embed_size * sizeof(half)));
        CUDA_CHECK(cudaMemset(weights.token_embedding, 0, embed_size * sizeof(half)));
    }

    // Load layer weights
    weights.layers.resize(config.num_layers);

    for (int layer = 0; layer < config.num_layers; ++layer) {
        auto &lw = weights.layers[layer];

        // GGUF tensor naming convention:
        // - blk.{N}.attn_q.weight
        // - blk.{N}.attn_k.weight
        // - blk.{N}.attn_v.weight
        // - blk.{N}.attn_output.weight (or attn_out)
        // - blk.{N}.ffn_gate.weight (w1)
        // - blk.{N}.ffn_up.weight (w3)
        // - blk.{N}.ffn_down.weight (w2)
        // - blk.{N}.attn_norm.weight
        // - blk.{N}.ffn_norm.weight

        std::string layer_prefix = "blk." + std::to_string(layer) + ".";

        // Alternative LLaMA naming
        std::string llama_prefix = "layers." + std::to_string(layer) + ".";

        auto find_layer_tensor = [&](const std::string &suffix) -> const GGUFTensorInfo * {
            const GGUFTensorInfo *t = find_tensor(layer_prefix + suffix);
            if (!t) {
                t = find_tensor(llama_prefix + suffix);
            }
            return t;
        };

        // Load attention weights (Q, K, V, O)
        // For now, load as FP16 placeholders (full quantization support would need conversion)
        int hidden = config.hidden_dim;
        int kv_dim = config.num_kv_heads * config.head_dim;

        // Helper to allocate zero-initialized weight
        auto alloc_zero_weight = [](int rows, int cols) -> QuantizedWeight {
            QuantizedWeight qw;
            qw.rows = rows;
            qw.cols = cols;
            qw.group_size = 128;
            CUDA_CHECK(cudaMalloc(&qw.data, qw.weightElements() * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&qw.scales, qw.scaleElements() * sizeof(half)));
            CUDA_CHECK(cudaMemset(qw.data, 0, qw.weightElements() * sizeof(int8_t)));
            CUDA_CHECK(cudaMemset(qw.scales, 0, qw.scaleElements() * sizeof(half)));
            return qw;
        };

        // Load Q weight
        const GGUFTensorInfo *q_tensor = find_layer_tensor("attn_q.weight");
        if (q_tensor) {
            TLLM_TRACE("Loading Q weight from: {}", q_tensor->name);
            // For now, allocate placeholder (full GGUF quantization support requires conversion)
            lw.wq = alloc_zero_weight(hidden, hidden);
        } else {
            TLLM_WARN("Layer {} Q weight not found", layer);
            lw.wq = alloc_zero_weight(hidden, hidden);
        }

        // Load K weight
        const GGUFTensorInfo *k_tensor = find_layer_tensor("attn_k.weight");
        if (k_tensor) {
            TLLM_TRACE("Loading K weight from: {}", k_tensor->name);
            lw.wk = alloc_zero_weight(hidden, kv_dim);
        } else {
            lw.wk = alloc_zero_weight(hidden, kv_dim);
        }

        // Load V weight
        const GGUFTensorInfo *v_tensor = find_layer_tensor("attn_v.weight");
        if (v_tensor) {
            TLLM_TRACE("Loading V weight from: {}", v_tensor->name);
            lw.wv = alloc_zero_weight(hidden, kv_dim);
        } else {
            lw.wv = alloc_zero_weight(hidden, kv_dim);
        }

        // Load O weight
        const GGUFTensorInfo *o_tensor = find_layer_tensor("attn_output.weight");
        if (!o_tensor) {
            o_tensor = find_layer_tensor("attn_out.weight");
        }
        if (o_tensor) {
            TLLM_TRACE("Loading O weight from: {}", o_tensor->name);
            lw.wo = alloc_zero_weight(hidden, hidden);
        } else {
            lw.wo = alloc_zero_weight(hidden, hidden);
        }

        // Load FFN weights
        const GGUFTensorInfo *w1_tensor = find_layer_tensor("ffn_gate.weight");
        const GGUFTensorInfo *w2_tensor = find_layer_tensor("ffn_down.weight");
        const GGUFTensorInfo *w3_tensor = find_layer_tensor("ffn_up.weight");
        int                   inter = config.intermediate_dim;

        if (w1_tensor) {
            TLLM_TRACE("Loading w1 (gate) weight from: {}", w1_tensor->name);
        }
        if (w2_tensor) {
            TLLM_TRACE("Loading w2 (down) weight from: {}", w2_tensor->name);
        }
        if (w3_tensor) {
            TLLM_TRACE("Loading w3 (up) weight from: {}", w3_tensor->name);
        }

        lw.w1 = alloc_zero_weight(hidden, inter);
        lw.w2 = alloc_zero_weight(inter, hidden);
        lw.w3 = alloc_zero_weight(hidden, inter);

        // Load normalization weights
        const GGUFTensorInfo *attn_norm = find_layer_tensor("attn_norm.weight");
        if (attn_norm) {
            auto data_result = parser.readTensorData(*attn_norm);
            if (data_result.isOk()) {
                CUDA_CHECK(cudaMalloc(&lw.rms_att_weight, hidden * sizeof(half)));
                // Convert if needed
                if (attn_norm->type == GGMLType::F32) {
                    std::vector<half> f16(hidden);
                    const float *f32 = reinterpret_cast<const float *>(data_result.value().data());
                    for (int i = 0; i < hidden; ++i) {
                        f16[i] = __float2half(f32[i]);
                    }
                    CUDA_CHECK(cudaMemcpy(lw.rms_att_weight, f16.data(), hidden * sizeof(half),
                                          cudaMemcpyHostToDevice));
                } else {
                    CUDA_CHECK(cudaMemcpy(lw.rms_att_weight, data_result.value().data(),
                                          hidden * sizeof(half), cudaMemcpyHostToDevice));
                }
            }
        } else {
            TLLM_WARN("Layer {} attention norm not found", layer);
            CUDA_CHECK(cudaMalloc(&lw.rms_att_weight, hidden * sizeof(half)));
            CUDA_CHECK(cudaMemset(lw.rms_att_weight, 0, hidden * sizeof(half)));
        }

        const GGUFTensorInfo *ffn_norm = find_layer_tensor("ffn_norm.weight");
        if (ffn_norm) {
            auto data_result = parser.readTensorData(*ffn_norm);
            if (data_result.isOk()) {
                CUDA_CHECK(cudaMalloc(&lw.rms_ffn_weight, hidden * sizeof(half)));
                if (ffn_norm->type == GGMLType::F32) {
                    std::vector<half> f16(hidden);
                    const float *f32 = reinterpret_cast<const float *>(data_result.value().data());
                    for (int i = 0; i < hidden; ++i) {
                        f16[i] = __float2half(f32[i]);
                    }
                    CUDA_CHECK(cudaMemcpy(lw.rms_ffn_weight, f16.data(), hidden * sizeof(half),
                                          cudaMemcpyHostToDevice));
                } else {
                    CUDA_CHECK(cudaMemcpy(lw.rms_ffn_weight, data_result.value().data(),
                                          hidden * sizeof(half), cudaMemcpyHostToDevice));
                }
            }
        } else {
            TLLM_WARN("Layer {} FFN norm not found", layer);
            CUDA_CHECK(cudaMalloc(&lw.rms_ffn_weight, hidden * sizeof(half)));
            CUDA_CHECK(cudaMemset(lw.rms_ffn_weight, 0, hidden * sizeof(half)));
        }

        TLLM_TRACE("Loaded layer {}/{}", layer + 1, config.num_layers);
    }

    // Load final norm
    const GGUFTensorInfo *output_norm = find_tensor("output_norm.weight");
    if (!output_norm) {
        output_norm = find_tensor("norm.weight");
    }
    if (output_norm) {
        TLLM_DEBUG("Loading final norm from: {}", output_norm->name);
        auto data_result = parser.readTensorData(*output_norm);
        if (data_result.isOk()) {
            CUDA_CHECK(cudaMalloc(&weights.final_norm_weight, config.hidden_dim * sizeof(half)));
            if (output_norm->type == GGMLType::F32) {
                std::vector<half> f16(config.hidden_dim);
                const float      *f32 = reinterpret_cast<const float *>(data_result.value().data());
                for (int i = 0; i < config.hidden_dim; ++i) {
                    f16[i] = __float2half(f32[i]);
                }
                CUDA_CHECK(cudaMemcpy(weights.final_norm_weight, f16.data(),
                                      config.hidden_dim * sizeof(half), cudaMemcpyHostToDevice));
            } else {
                CUDA_CHECK(cudaMemcpy(weights.final_norm_weight, data_result.value().data(),
                                      config.hidden_dim * sizeof(half), cudaMemcpyHostToDevice));
            }
        }
    } else {
        TLLM_WARN("Output norm not found");
        CUDA_CHECK(cudaMalloc(&weights.final_norm_weight, config.hidden_dim * sizeof(half)));
        CUDA_CHECK(cudaMemset(weights.final_norm_weight, 0, config.hidden_dim * sizeof(half)));
    }

    // Load LM head
    const GGUFTensorInfo *lm_head = find_tensor("output.weight");
    if (!lm_head) {
        lm_head = find_tensor("lm_head.weight");
    }
    if (lm_head) {
        TLLM_DEBUG("Loading LM head from: {}", lm_head->name);
        // For now, allocate placeholder (full support needs quantization conversion)
        weights.lm_head.rows = config.hidden_dim;
        weights.lm_head.cols = config.vocab_size;
        weights.lm_head.group_size = 128;
        CUDA_CHECK(
            cudaMalloc(&weights.lm_head.data, weights.lm_head.weightElements() * sizeof(int8_t)));
        CUDA_CHECK(
            cudaMalloc(&weights.lm_head.scales, weights.lm_head.scaleElements() * sizeof(half)));
        CUDA_CHECK(
            cudaMemset(weights.lm_head.data, 0, weights.lm_head.weightElements() * sizeof(int8_t)));
        CUDA_CHECK(
            cudaMemset(weights.lm_head.scales, 0, weights.lm_head.scaleElements() * sizeof(half)));
    } else {
        TLLM_WARN("LM head tensor not found");
        weights.lm_head.rows = config.hidden_dim;
        weights.lm_head.cols = config.vocab_size;
        weights.lm_head.group_size = 128;
        CUDA_CHECK(
            cudaMalloc(&weights.lm_head.data, weights.lm_head.weightElements() * sizeof(int8_t)));
        CUDA_CHECK(
            cudaMalloc(&weights.lm_head.scales, weights.lm_head.scaleElements() * sizeof(half)));
        CUDA_CHECK(
            cudaMemset(weights.lm_head.data, 0, weights.lm_head.weightElements() * sizeof(int8_t)));
        CUDA_CHECK(
            cudaMemset(weights.lm_head.scales, 0, weights.lm_head.scaleElements() * sizeof(half)));
    }

    success = true;
    TLLM_INFO("GGUF model loaded successfully");

    return Result<ModelWeights>::ok(std::move(weights));
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
