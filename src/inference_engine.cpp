#include "tiny_llm/inference_engine.h"
#include "elementwise.cuh"
#include "rmsnorm.cuh"
#include "rope.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/execution_common.h"
#include "tiny_llm/logger.h"
#include "tiny_llm/model_loader.h"
#include "tiny_llm/validator.h"
#include "w8a16_matmul.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>

namespace tiny_llm {

namespace {

// Thread-local RNG shared by all sampling helpers.  Previous code constructed
// a fresh std::mt19937 (plus a std::discrete_distribution) for every token,
// which allocates and initializes O(vocab_size) state per decode step.
std::mt19937 &samplingRng(unsigned seed) {
    static thread_local std::mt19937 gen(std::random_device{}());
    if (seed != 0) {
        gen.seed(seed);
    }
    return gen;
}

// Sample an index from an already-normalized probability vector using the
// cumulative-distribution method (O(log n) after O(n) prefix sums).  This
// avoids the expensive std::discrete_distribution constructor.
template <typename Probs, typename Indices>
int sampleFromCdf(Probs &probs, Indices &indices, int count, std::mt19937 &gen) {
    std::vector<float> cdf(count);
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        sum += probs[i];
        cdf[i] = sum;
    }
    if (sum <= 0.0f) {
        return indices[0];
    }
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(gen);
    auto it = std::lower_bound(cdf.begin(), cdf.begin() + count, r);
    int idx = static_cast<int>(it - cdf.begin());
    return indices[std::min(idx, count - 1)];
}

} // namespace

Result<std::unique_ptr<InferenceEngine>> InferenceEngine::load(const std::string &model_path,
                                                               const ModelConfig &config) {
    TLLM_INFO("Loading model from: {}", model_path);

    // Validate model config
    auto config_result = Validator::validateModelConfig(config);
    if (config_result.isErr()) {
        TLLM_ERROR("Invalid model config: {}", config_result.error());
        return Result<std::unique_ptr<InferenceEngine>>::err("Invalid model config: " +
                                                             config_result.error());
    }

    if (model_path.size() >= 5 && model_path.substr(model_path.size() - 5) == ".gguf") {
        // GGUF 运行时加载：从文件提取配置并转换权重到 W8A16
        ModelConfig gguf_config = config;
        auto        result      = ModelLoader::loadGGUF(model_path, gguf_config);
        if (result.isErr()) {
            TLLM_ERROR("Failed to load GGUF model: {}", result.error());
            return Result<std::unique_ptr<InferenceEngine>>::err(result.error());
        }
        auto engine = std::make_unique<InferenceEngine>(gguf_config, std::move(result.value()));
        TLLM_INFO("GGUF model loaded: hidden_dim={}, num_layers={}, vocab_size={}",
                  gguf_config.hidden_dim, gguf_config.num_layers, gguf_config.vocab_size);
        return Result<std::unique_ptr<InferenceEngine>>::ok(std::move(engine));
    }

    // Load model weights
    auto result = ModelLoader::loadBin(model_path, config);
    if (result.isErr()) {
        TLLM_ERROR("Failed to load model: {}", result.error());
        return Result<std::unique_ptr<InferenceEngine>>::err(result.error());
    }

    auto engine = std::make_unique<InferenceEngine>(config, std::move(result.value()));

    TLLM_INFO("Model loaded successfully: hidden_dim={}, num_layers={}, vocab_size={}",
              config.hidden_dim, config.num_layers, config.vocab_size);

    return Result<std::unique_ptr<InferenceEngine>>::ok(std::move(engine));
}

InferenceEngine::InferenceEngine(const ModelConfig &config, ModelWeights &&weights)
    : config_(config), weights_(std::move(weights)) {

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));

    // Initialize KV cache
    KVCacheConfig kv_config;
    kv_config.num_layers = config_.num_layers;
    kv_config.num_kv_heads = config_.num_kv_heads;
    kv_config.head_dim = config_.head_dim;
    kv_config.max_seq_len = config_.max_seq_len;
    kv_config.max_batch_size = 1;

    auto kv_cache_result = KVCacheManager::create(kv_config);
    if (kv_cache_result.isErr()) {
        throw std::runtime_error("Failed to create KV cache: " + kv_cache_result.error());
    }
    kv_cache_ = std::move(kv_cache_result.value());

    // Create transformer layers
    // 共享中间激活工作区：所有层复用，避免按层数线性放大显存
    workspace_.allocate(config_);

    layers_.reserve(config_.num_layers);
    for (int i = 0; i < config_.num_layers; ++i) {
        layers_.push_back(
            std::make_unique<TransformerLayer>(i, weights_.layers[i], config_, &workspace_));
    }

    // Allocate buffers
    size_t hidden_size = config_.max_seq_len * config_.hidden_dim * sizeof(half);
    size_t logits_size = config_.vocab_size * sizeof(half);

    CUDA_CHECK(cudaMalloc(&hidden_states_, hidden_size));
    CUDA_CHECK(cudaMalloc(&logits_, logits_size));
    // 任务 3.1：decode 可见 KV 长度的 device int 缓冲（CUDA Graph 前置）。
    decode_len_ = DeviceBuffer<int>(1);
    // 任务 3.2：decode 固定输入缓冲（token id + RoPE 起始位置）。
    graph_token_ = DeviceBuffer<int>(1);
    rope_pos_ = DeviceBuffer<int>(1);
    if (std::getenv("TLLM_DEBUG_ZERO")) {
        CUDA_CHECK(cudaMemset(hidden_states_, 0, hidden_size));
        CUDA_CHECK(cudaMemset(logits_, 0, logits_size));
    }

    // 任务 C2：CUDA Graphs 默认开启；TLLM_CUDA_GRAPHS=0 显式关闭（opt-out）。
    // 状态仍打印当前值。
    if (const char *g = std::getenv("TLLM_CUDA_GRAPHS"); g && std::string(g) == "0") {
        cuda_graphs_enabled_ = false;
        TLLM_INFO("CUDA Graphs: decode graph capture/replay DISABLED (TLLM_CUDA_GRAPHS=0)");
    } else {
        cuda_graphs_enabled_ = true;
        TLLM_INFO("CUDA Graphs: decode graph capture/replay ENABLED (default; set TLLM_CUDA_GRAPHS=0 to disable)");
    }

    // TLLM-003: Allocate and precompute RoPE cos/sin half cache
    int half_d = config_.head_dim / 2;
    size_t rope_cache_size = static_cast<size_t>(config_.max_seq_len) * half_d * sizeof(float);
    CUDA_CHECK(cudaMalloc(&rope_cos_, rope_cache_size));
    CUDA_CHECK(cudaMalloc(&rope_sin_, rope_cache_size));
    kernels::rope_precompute_cache(rope_cos_, rope_sin_, config_.max_seq_len, config_.head_dim,
                                   config_.rope_theta, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

InferenceEngine::~InferenceEngine() {
    layers_.clear();
    kv_cache_.reset();

    // 任务 3.2：释放 CUDA Graph（exec 与 graph 对象）。
    if (decode_graph_exec_) {
        cudaError_t err = cudaGraphExecDestroy(decode_graph_exec_);
        if (err != cudaSuccess) {
            TLLM_ERROR("CUDA error destroying decode graph exec: {}", cudaGetErrorString(err));
        }
        decode_graph_exec_ = nullptr;
    }
    if (decode_graph_) {
        cudaError_t err = cudaGraphDestroy(decode_graph_);
        if (err != cudaSuccess) {
            TLLM_ERROR("CUDA error destroying decode graph: {}", cudaGetErrorString(err));
        }
        decode_graph_ = nullptr;
    }

    if (hidden_states_) {
        cudaError_t err = cudaFree(hidden_states_);
        if (err != cudaSuccess) {
            TLLM_ERROR("CUDA error freeing hidden_states: {}", cudaGetErrorString(err));
        }
        hidden_states_ = nullptr;
    }
    if (logits_) {
        cudaError_t err = cudaFree(logits_);
        if (err != cudaSuccess) {
            TLLM_ERROR("CUDA error freeing logits: {}", cudaGetErrorString(err));
        }
        logits_ = nullptr;
    }
    // TLLM-003: Free RoPE cache
    if (rope_cos_) {
        cudaError_t err = cudaFree(rope_cos_);
        if (err != cudaSuccess) {
            TLLM_ERROR("CUDA error freeing rope_cos: {}", cudaGetErrorString(err));
        }
        rope_cos_ = nullptr;
    }
    if (rope_sin_) {
        cudaError_t err = cudaFree(rope_sin_);
        if (err != cudaSuccess) {
            TLLM_ERROR("CUDA error freeing rope_sin: {}", cudaGetErrorString(err));
        }
        rope_sin_ = nullptr;
    }

    if (stream_) {
        cudaError_t err = cudaStreamDestroy(stream_);
        if (err != cudaSuccess) {
            TLLM_ERROR("CUDA error destroying stream: {}", cudaGetErrorString(err));
        }
        stream_ = nullptr;
    }

    ModelLoader::freeWeights(weights_);
}

Result<std::vector<int>> InferenceEngine::generate(const std::vector<int> &prompt_tokens,
                                                   const GenerationConfig &config) {
    stats_ = GenerationStats{};

    // 1. Validate generation config
    auto config_result = config.validate();
    if (config_result.isErr()) {
        TLLM_ERROR("Invalid generation config: {}", config_result.error());
        return Result<std::vector<int>>::err("Invalid generation config: " + config_result.error());
    }

    // 2. Validate prompt tokens
    if (prompt_tokens.empty()) {
        TLLM_ERROR("generate: prompt_tokens is empty");
        return Result<std::vector<int>>::err("prompt_tokens cannot be empty");
    }

    auto token_result =
        Validator::validateTokenSequence(prompt_tokens, config_.vocab_size, "generate");
    if (token_result.isErr()) {
        TLLM_ERROR("{}", token_result.error());
        return Result<std::vector<int>>::err(token_result.error());
    }

    // 3. Validate prompt length
    auto length_result = Validator::validatePromptLength(
        static_cast<int>(prompt_tokens.size()), config.max_new_tokens, config_.max_seq_len);
    if (length_result.isErr()) {
        TLLM_ERROR("{}", length_result.error());
        return Result<std::vector<int>>::err(length_result.error());
    }

    stats_.prompt_tokens = static_cast<int>(prompt_tokens.size());
    TLLM_INFO("Starting generation: prompt_tokens={}, max_new_tokens={}", prompt_tokens.size(),
              config.max_new_tokens);

    // Allocate sequence in KV cache
    int  total_len = static_cast<int>(prompt_tokens.size()) + config.max_new_tokens;
    auto alloc_result = kv_cache_->allocateSequence(std::min(total_len, config_.max_seq_len));
    if (alloc_result.isErr()) {
        TLLM_ERROR("Failed to allocate KV cache: {}", alloc_result.error());
        return Result<std::vector<int>>::err("Failed to allocate KV cache: " +
                                             alloc_result.error());
    }
    int seq_id = alloc_result.value();
    TLLM_DEBUG("Allocated KV cache sequence: seq_id={}", seq_id);

    std::vector<int> output_tokens;
    output_tokens.reserve(config.max_new_tokens);

    // Prefill phase
    auto prefill_start = std::chrono::high_resolution_clock::now();
    auto prefill_result = prefill(prompt_tokens, seq_id);
    if (prefill_result.isErr()) {
        TLLM_ERROR("generate: prefill failed: {}", prefill_result.error());
        kv_cache_->releaseSequence(seq_id);
        return Result<std::vector<int>>::err("Prefill failed: " + prefill_result.error());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto prefill_end = std::chrono::high_resolution_clock::now();
    stats_.prefill_time_ms =
        std::chrono::duration<float, std::milli>(prefill_end - prefill_start).count();

    TLLM_DEBUG("Prefill completed: time={:.2f}ms", stats_.prefill_time_ms);

    // Decode phase
    auto decode_start = std::chrono::high_resolution_clock::now();
    int  position = static_cast<int>(prompt_tokens.size());
    int  prev_token = prompt_tokens.empty() ? config_.bos_token_id : prompt_tokens.back();
    int  generated = 0;
    // repetition_penalty 依赖"已见 token"集合（prompt + 已生成）
    std::vector<int> past_tokens = prompt_tokens;

    if (!prompt_tokens.empty() && position > 0 && generated < config.max_new_tokens) {
        half *last_hidden = hidden_states_ + (position - 1) * config_.hidden_dim;
        int   next_token = sampleFromHidden(last_hidden, config, past_tokens);
        output_tokens.push_back(next_token);
        past_tokens.push_back(next_token);
        ++generated;

        if (next_token == config_.eos_token_id) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto decode_end = std::chrono::high_resolution_clock::now();
            stats_.decode_time_ms =
                std::chrono::duration<float, std::milli>(decode_end - decode_start).count();
            stats_.tokens_generated = static_cast<int>(output_tokens.size());
            if (stats_.decode_time_ms > 0) {
                stats_.tokens_per_second =
                    stats_.tokens_generated / (stats_.decode_time_ms / 1000.0f);
            }
            TLLM_INFO("Generation stopped at EOS token: generated={}, time={:.2f}ms",
                      stats_.tokens_generated, stats_.decode_time_ms);
            auto release_result = kv_cache_->releaseSequence(seq_id);
            if (release_result.isErr()) {
                TLLM_WARN("Failed to release sequence: {}", release_result.error());
            }
            return Result<std::vector<int>>::ok(output_tokens);
        }
        prev_token = next_token;
    }

    while (generated < config.max_new_tokens && position < config_.max_seq_len) {
        auto decode_result = decodeStep(seq_id, position, prev_token, config, past_tokens);
        if (decode_result.isErr()) {
            TLLM_ERROR("generate: decodeStep failed at position {}: {}", position,
                       decode_result.error());
            kv_cache_->releaseSequence(seq_id);
            return Result<std::vector<int>>::err("Decode failed: " + decode_result.error());
        }
        int next_token = decode_result.value();
        output_tokens.push_back(next_token);
        past_tokens.push_back(next_token);
        ++generated;

        // Check for EOS
        if (next_token == config_.eos_token_id) break;

        prev_token = next_token;
        ++position;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto decode_end = std::chrono::high_resolution_clock::now();
    stats_.decode_time_ms =
        std::chrono::duration<float, std::milli>(decode_end - decode_start).count();
    stats_.tokens_generated = static_cast<int>(output_tokens.size());

    if (stats_.decode_time_ms > 0) {
        stats_.tokens_per_second = stats_.tokens_generated / (stats_.decode_time_ms / 1000.0f);
    }

    TLLM_INFO("Generation completed: tokens={}, prefill={:.2f}ms, decode={:.2f}ms, tps={:.2f}",
              stats_.tokens_generated, stats_.prefill_time_ms, stats_.decode_time_ms,
              stats_.tokens_per_second);

    // Release sequence
    auto release_result = kv_cache_->releaseSequence(seq_id);
    if (release_result.isErr()) {
        TLLM_WARN("Failed to release sequence: {}", release_result.error());
    }

    return Result<std::vector<int>>::ok(output_tokens);
}

Result<void> InferenceEngine::prefill(const std::vector<int> &tokens, int seq_id) {
    int num_tokens = static_cast<int>(tokens.size());
    if (num_tokens <= 0) {
        TLLM_WARN("prefill: empty token sequence");
        return Result<void>::err("prefill: empty token sequence");
    }

    TLLM_TRACE("prefill: seq_id={}, num_tokens={}", seq_id, num_tokens);

    // Validate token IDs (debug mode only for performance)
    TLLM_DEBUG_IF(true, "prefill: validating {} token IDs", num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        if (tokens[i] < 0 || tokens[i] >= config_.vocab_size) {
            TLLM_ERROR("prefill: invalid token_id {} at position {}, vocab_size={}", tokens[i], i,
                       config_.vocab_size);
            return Result<void>::err("prefill: invalid token_id " + std::to_string(tokens[i]) +
                                     " at position " + std::to_string(i));
        }
    }

    // Embed tokens
    std::vector<int>  h_tokens(tokens);
    DeviceBuffer<int> d_tokens(num_tokens);
    d_tokens.copyFromHost(h_tokens.data(), num_tokens, stream_);

    embedTokens(d_tokens.data(), num_tokens, hidden_states_);

    // 任务 3.2：prefill 起始位置 0、append 写位置 = 当前可见长度（0）。
    // transformer 层从 device 读这两个值（graph 重放前置）。
    const int zero_pos = 0;
    rope_pos_.copyFromHost(&zero_pos, 1, stream_);
    kv_cache_->setAppendPos(0, stream_);

    // Forward through all layers
    for (auto &layer : layers_) {
        auto layer_result = layer->forwardPrefill(hidden_states_, *kv_cache_, seq_id, num_tokens,
                                                   rope_pos_.data(), rope_cos_, rope_sin_, stream_);
        if (layer_result.isErr()) {
            TLLM_ERROR("prefill: layer {} failed: {}", layer->getLayerIdx(),
                       layer_result.error());
            return layer_result;
        }
    }

    auto advance_result = kv_cache_->advanceSeqLen(seq_id, num_tokens);
    if (advance_result.isErr()) {
        TLLM_ERROR("prefill: {}", advance_result.error());
        return advance_result;
    }

    return Result<void>::ok();
}

Result<int> InferenceEngine::decodeStep(int seq_id, int position, int token_id,
                                          const GenerationConfig &config,
                                          const std::vector<int> &past_tokens) {
    // 任务 3.2：decode 固定使用 hidden_states_ 的最后一行（不随 position 变），
    // 使 graph 捕获的地址可无更新重放。该行只在单次 decode step 内使用，
    // prefill 行（0..num_tokens-1）不受影响。
    half *token_state =
        hidden_states_ + static_cast<size_t>(config_.max_seq_len - 1) * config_.hidden_dim;

    // 写入 device 端输入/参数（与后续 kernel 同一 stream，保证顺序）：
    //  - graph_token_: 待 decode 的 token id（embed 输入，地址跨重放稳定）
    //  - decode_len_ : appendKV 后的可见长度 = getSeqLen + 1（attention）
    //  - rope_pos_   : 本 token 绝对位置（RoPE）
    //  - append_pos_ : append 写位置 = 当前可见长度（KV cache）
    graph_token_.copyFromHost(&token_id, 1, stream_);
    const int decode_len = kv_cache_->getSeqLen(seq_id) + 1;
    decode_len_.copyFromHost(&decode_len, 1, stream_);
    rope_pos_.copyFromHost(&position, 1, stream_);
    kv_cache_->setAppendPos(kv_cache_->getSeqLen(seq_id), stream_);

    if (cuda_graphs_enabled_ && graph_captured_) {
        // 已捕获：重放 graph（第 2 步起）。
        CUDA_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_));
    } else if (cuda_graphs_enabled_ && !graph_captured_) {
        // 第一次 decode：先直接执行并同步，再在 capture 区内记录一次
        //（capture 不真正执行 kernel，只记录节点）。本步结果来自直接执行；
        // graph 从第 2 步起重放。任何失败 → 关闭 graphs 回退常规路径。
        auto direct_result = runDecodeDevicePath(token_state, seq_id, position);
        if (direct_result.isErr()) {
            return Result<int>::err(direct_result.error());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        bool capture_ok = false;
        try {
            cudaError_t begin_err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
            if (begin_err == cudaSuccess) {
                auto capture_result = runDecodeDevicePath(token_state, seq_id, position);
                cudaError_t end_err = cudaStreamEndCapture(stream_, &decode_graph_);
                capture_ok = capture_result.isOk() && end_err == cudaSuccess;
                if (!capture_ok) {
                    if (decode_graph_) {
                        cudaGraphDestroy(decode_graph_);
                        decode_graph_ = nullptr;
                    }
                    TLLM_WARN("CUDA Graphs: capture failed ({}), falling back",
                              end_err == cudaSuccess ? capture_result.error()
                                                     : cudaGetErrorString(end_err));
                }
            } else {
                TLLM_WARN("CUDA Graphs: cudaStreamBeginCapture failed ({}), falling back",
                          cudaGetErrorString(begin_err));
            }
        } catch (const CudaException &e) {
            TLLM_WARN("CUDA Graphs: capture threw ({}), falling back", e.what());
            if (decode_graph_) {
                cudaGraphDestroy(decode_graph_);
                decode_graph_ = nullptr;
            }
            capture_ok = false;
        }

        if (capture_ok) {
            cudaError_t inst_err = cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr,
                                                        nullptr, 0);
            if (inst_err != cudaSuccess) {
                TLLM_WARN("CUDA Graphs: cudaGraphInstantiate failed ({}), falling back",
                          cudaGetErrorString(inst_err));
                cudaGraphDestroy(decode_graph_);
                decode_graph_ = nullptr;
                cuda_graphs_enabled_ = false;
            } else {
                graph_captured_ = true;
                TLLM_INFO("CUDA Graphs: decode graph captured & instantiated");
            }
        } else {
            cuda_graphs_enabled_ = false;
        }
        // 本步结果来自上面的直接执行，不重复 launch。
    } else {
        // 未启用 graphs，或捕获失败后的回退路径。
        auto direct_result = runDecodeDevicePath(token_state, seq_id, position);
        if (direct_result.isErr()) {
            return Result<int>::err(direct_result.error());
        }
    }

    auto advance_result = kv_cache_->advanceSeqLen(seq_id, 1);
    if (advance_result.isErr()) {
        TLLM_ERROR("decodeStep: {}", advance_result.error());
        return Result<int>::err(advance_result.error());
    }

    return Result<int>::ok(sampleFromLogits(config, past_tokens));
}

// 任务 3.2：decode 的确定性 device 序列 —— graph 捕获与直接执行共用。
// 只包含 kernel/memcpy（可捕获）；advanceSeqLen（host）、logits D2H、采样
// 不在此列。
Result<void> InferenceEngine::runDecodeDevicePath(half *token_state, int seq_id, int position) {
    embedTokens(graph_token_.data(), 1, token_state);

    for (auto &layer : layers_) {
        auto layer_result = layer->forward(token_state, *kv_cache_, seq_id, position,
                                            decode_len_.data(), rope_pos_.data(), rope_cos_,
                                            rope_sin_, stream_);
        if (layer_result.isErr()) {
            TLLM_ERROR("decodeStep: layer {} failed: {}", layer->getLayerIdx(),
                       layer_result.error());
            return Result<void>::err(layer_result.error());
        }
    }

    // 任务 4.3：final norm + lm_head 走共享 helper（与 FFI 路径一致）
    finalNormAndComputeLogits(token_state, weights_, config_, logits_, stream_);
    return Result<void>::ok();
}

// 任务 3.2：从 logits_ 采样（graph 不覆盖的 host 侧部分）。
int InferenceEngine::sampleFromLogits(const GenerationConfig &config,
                                      const std::vector<int> &past_tokens) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<half> h_logits(config_.vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), logits_, config_.vocab_size * sizeof(half),
                          cudaMemcpyDeviceToHost));

    return sample(h_logits.data(), config, past_tokens);
}

int InferenceEngine::sampleFromHidden(half *hidden_state, const GenerationConfig &config,
                                      const std::vector<int> &past_tokens) {
    // 任务 4.3：final norm + lm_head 走共享 helper（与 FFI 路径一致）
    finalNormAndComputeLogits(hidden_state, weights_, config_, logits_, stream_);
    return sampleFromLogits(config, past_tokens);
}

void InferenceEngine::embedTokens(const int *tokens, int num_tokens, half *output) {
    kernels::gather_embeddings(tokens, weights_.token_embedding, output, num_tokens,
                               config_.hidden_dim, config_.vocab_size, stream_);
}

void InferenceEngine::computeLogits(const half *hidden_states, int num_tokens, half *logits) {
    // LM head projection: hidden_states @ lm_head.T
    // 只做 lm_head，不包含 final RMSNorm（RMSNorm 由调用方/helper 负责）。
    // 优先 FP16 lm_head（output 层不量化，保持 logits 精度与 llama.cpp 对齐）；
    // W8A16 版本作为后备。
    // 任务 C1：M==1（decode）时传转置布局走 coalesced 快路径；M>1 自动回退。
    if (weights_.lm_head_fp16) {
        kernels::fp16_matmul(hidden_states, weights_.lm_head_fp16, weights_.lm_head_fp16_t,
                             logits, num_tokens, config_.vocab_size, config_.hidden_dim, stream_);
    } else if (weights_.lm_head.isValid()) {
        kernels::w8a16_matmul(hidden_states, weights_.lm_head.data, weights_.lm_head.scales,
                              weights_.lm_head.data_t, weights_.lm_head.scales_t, logits,
                              num_tokens, config_.vocab_size, config_.hidden_dim,
                              weights_.lm_head.group_size, stream_);
    }
}

void InferenceEngine::finalNorm(const half *input, half *output, int num_tokens) {
    if (weights_.final_norm_weight) {
        kernels::rmsnorm(input, weights_.final_norm_weight, output, num_tokens, config_.hidden_dim,
                         config_.rms_norm_eps, stream_);
    }
}

void InferenceEngine::applyRepetitionPenalty(half *logits, const std::vector<int> &past_tokens,
                                            float penalty, int vocab_size) {
    if (penalty == 1.0f || past_tokens.empty() || logits == nullptr || vocab_size <= 0) {
        return;
    }
    for (int id : past_tokens) {
        if (id < 0 || id >= vocab_size) continue;
        float v = __half2float(logits[id]);
        v = (v < 0.0f) ? v * penalty : v / penalty;
        logits[id] = __float2half(v);
    }
}

int InferenceEngine::sample(const half *logits, const GenerationConfig &config,
                            const std::vector<int> &past_tokens) {
    // repetition_penalty（llama.cpp 语义）：对"已见 token"的 logit 施加惩罚，
    // 负 logit 乘 penalty、正 logit 除 penalty，再走采样分发。penalty==1.0
    // 或无可查历史时零开销直通。
    const half *effective = logits;
    std::vector<half> penalized;
    if (config.repetition_penalty != 1.0f && !past_tokens.empty()) {
        penalized.assign(logits, logits + config_.vocab_size);
        applyRepetitionPenalty(penalized.data(), past_tokens, config.repetition_penalty,
                               config_.vocab_size);
        effective = penalized.data();
    }

    if (!config.do_sample) {
        return sampleGreedy(effective, config_.vocab_size);
    }

    if (config.top_p < 1.0f) {
        return sampleTopP(effective, config_.vocab_size, config.top_p, config.temperature);
    }

    if (config.top_k > 0) {
        return sampleTopK(effective, config_.vocab_size, config.top_k, config.temperature);
    }

    return sampleTemperature(effective, config_.vocab_size, config.temperature);
}

// Greedy sampling: return argmax
int InferenceEngine::sampleGreedy(const half *logits, int vocab_size) {
    if (!logits || vocab_size <= 0) {
        return 0;
    }

    int   max_idx = 0;
    float max_val = __half2float(logits[0]);

    for (int i = 1; i < vocab_size; ++i) {
        float val = __half2float(logits[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    return max_idx;
}

// Temperature sampling
int InferenceEngine::sampleTemperature(const half *logits, int vocab_size, float temperature,
                                       unsigned seed) {
    if (!logits || vocab_size <= 0) {
        return 0;
    }
    temperature = std::max(temperature, 1e-5f);

    std::vector<float> probs(vocab_size);
    float              max_logit = __half2float(logits[0]);

    // Find max for numerical stability
    for (int i = 1; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, __half2float(logits[i]));
    }

    // Apply temperature and softmax
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((__half2float(logits[i]) - max_logit) / temperature);
        sum += probs[i];
    }

    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    // Sample from the distribution via CDF lookup.
    std::mt19937                    &gen = samplingRng(seed);
    std::vector<int>                indices(vocab_size);
    for (int i = 0; i < vocab_size; ++i) indices[i] = i;
    return sampleFromCdf(probs, indices, vocab_size, gen);
}

// Top-k sampling
int InferenceEngine::sampleTopK(const half *logits, int vocab_size, int k, float temperature,
                                unsigned seed) {
    if (!logits || vocab_size <= 0) {
        return 0;
    }
    temperature = std::max(temperature, 1e-5f);
    k = std::max(1, std::min(k, vocab_size));

    // Get top-k indices
    std::vector<std::pair<float, int>> logit_pairs(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        logit_pairs[i] = {__half2float(logits[i]), i};
    }

    std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

    // Apply temperature and softmax to top-k
    std::vector<float> probs(k);
    float              max_logit = logit_pairs[0].first;
    float              sum = 0.0f;

    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp((logit_pairs[i].first - max_logit) / temperature);
        sum += probs[i];
    }

    for (int i = 0; i < k; ++i) {
        probs[i] /= sum;
    }

    // Sample (CDF lookup; logit_pairs[sampled].second is the token id).
    std::mt19937     &gen = samplingRng(seed);
    std::vector<int>  indices(k);
    for (int i = 0; i < k; ++i) indices[i] = i;
    int pick = sampleFromCdf(probs, indices, k, gen);
    return logit_pairs[pick].second;
}

// Top-p (nucleus) sampling
int InferenceEngine::sampleTopP(const half *logits, int vocab_size, float p, float temperature,
                                unsigned seed) {
    if (!logits || vocab_size <= 0) {
        return 0;
    }
    temperature = std::max(temperature, 1e-5f);
    p = std::min(std::max(p, 1e-5f), 1.0f);

    // Sort by logit value
    std::vector<std::pair<float, int>> logit_pairs(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        logit_pairs[i] = {__half2float(logits[i]), i};
    }

    std::sort(logit_pairs.begin(), logit_pairs.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    // Apply temperature and compute cumulative probability
    float              max_logit = logit_pairs[0].first;
    std::vector<float> probs(vocab_size);
    float              sum = 0.0f;

    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logit_pairs[i].first - max_logit) / temperature);
        sum += probs[i];
    }

    // Normalize and find cutoff
    float cumsum = 0.0f;
    int   cutoff = vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
        cumsum += probs[i];
        if (cumsum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    // Renormalize top-p tokens
    float top_p_sum = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        top_p_sum += probs[i];
    }
    for (int i = 0; i < cutoff; ++i) {
        probs[i] /= top_p_sum;
    }

    // Sample (CDF lookup over the truncated top-p set).
    std::mt19937     &gen = samplingRng(seed);
    std::vector<int>  indices(cutoff);
    for (int i = 0; i < cutoff; ++i) indices[i] = i;
    int pick = sampleFromCdf(probs, indices, cutoff, gen);
    return logit_pairs[pick].second;
}

} // namespace tiny_llm
