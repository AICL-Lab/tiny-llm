#include "tiny_llm/inference_engine.h"
#include "elementwise.cuh"
#include "rmsnorm.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/model_loader.h"
#include "w8a16_matmul.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

namespace tiny_llm {

Result<std::unique_ptr<InferenceEngine>>
InferenceEngine::load(const std::string &model_path,
                      const ModelConfig &config) {
  if (model_path.size() >= 5 &&
      model_path.substr(model_path.size() - 5) == ".gguf") {
    return Result<std::unique_ptr<InferenceEngine>>::err(
        "GGUF runtime loading is not supported yet; use the test binary format loaded via loadBin().");
  }

  // Load model weights
  auto result = ModelLoader::loadBin(model_path, config);
  if (result.isErr()) {
    return Result<std::unique_ptr<InferenceEngine>>::err(result.error());
  }

  auto engine =
      std::make_unique<InferenceEngine>(config, std::move(result.value()));
  return Result<std::unique_ptr<InferenceEngine>>::ok(std::move(engine));
}

InferenceEngine::InferenceEngine(const ModelConfig &config,
                                 ModelWeights &&weights)
    : config_(config), weights_(std::move(weights)) {

  // Create CUDA stream
  CUDA_CHECK(cudaStreamCreate(&stream_));

  // Initialize KV cache
  KVCacheConfig kv_config;
  kv_config.num_layers = config_.num_layers;
  kv_config.num_heads = config_.num_kv_heads;
  kv_config.head_dim = config_.head_dim;
  kv_config.max_seq_len = config_.max_seq_len;
  kv_config.max_batch_size = 1;
  kv_cache_ = std::make_unique<KVCacheManager>(kv_config);

  // Create transformer layers
  layers_.reserve(config_.num_layers);
  for (int i = 0; i < config_.num_layers; ++i) {
    layers_.push_back(
        std::make_unique<TransformerLayer>(i, weights_.layers[i], config_));
  }

  // Allocate buffers
  size_t hidden_size = config_.max_seq_len * config_.hidden_dim * sizeof(half);
  size_t logits_size = config_.vocab_size * sizeof(half);

  CUDA_CHECK(cudaMalloc(&hidden_states_, hidden_size));
  CUDA_CHECK(cudaMalloc(&logits_, logits_size));
}

InferenceEngine::~InferenceEngine() {
  layers_.clear();
  kv_cache_.reset();

  if (hidden_states_) {
    cudaFree(hidden_states_);
    hidden_states_ = nullptr;
  }
  if (logits_) {
    cudaFree(logits_);
    logits_ = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }

  ModelLoader::freeWeights(weights_);
}

std::vector<int>
InferenceEngine::generate(const std::vector<int> &prompt_tokens,
                          const GenerationConfig &config) {
  stats_ = GenerationStats{};
  stats_.prompt_tokens = static_cast<int>(prompt_tokens.size());

  // Allocate sequence in KV cache
  int total_len =
      static_cast<int>(prompt_tokens.size()) + config.max_new_tokens;
  auto alloc_result =
      kv_cache_->allocateSequence(std::min(total_len, config_.max_seq_len));
  if (alloc_result.isErr()) {
    return {}; // Failed to allocate
  }
  int seq_id = alloc_result.value();

  std::vector<int> output_tokens;
  output_tokens.reserve(config.max_new_tokens);

  // Prefill phase
  auto prefill_start = std::chrono::high_resolution_clock::now();
  prefill(prompt_tokens, seq_id);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  auto prefill_end = std::chrono::high_resolution_clock::now();
  stats_.prefill_time_ms =
      std::chrono::duration<float, std::milli>(prefill_end - prefill_start)
          .count();

  // Decode phase
  auto decode_start = std::chrono::high_resolution_clock::now();
  int position = static_cast<int>(prompt_tokens.size());
  int prev_token =
      prompt_tokens.empty() ? config_.bos_token_id : prompt_tokens.back();
  int generated = 0;

  if (!prompt_tokens.empty() && position > 0 &&
      generated < config.max_new_tokens) {
    half *last_hidden = hidden_states_ + (position - 1) * config_.hidden_dim;
    int next_token = sampleFromHidden(last_hidden, config);
    output_tokens.push_back(next_token);
    ++generated;

    if (next_token == config_.eos_token_id) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      auto decode_end = std::chrono::high_resolution_clock::now();
      stats_.decode_time_ms =
          std::chrono::duration<float, std::milli>(decode_end - decode_start)
              .count();
      stats_.tokens_generated = static_cast<int>(output_tokens.size());
      if (stats_.decode_time_ms > 0) {
        stats_.tokens_per_second =
            stats_.tokens_generated / (stats_.decode_time_ms / 1000.0f);
      }
      kv_cache_->releaseSequence(seq_id);
      return output_tokens;
    }
    prev_token = next_token;
  }

  while (generated < config.max_new_tokens && position < config_.max_seq_len) {
    int next_token = decodeStep(seq_id, position, prev_token, config);
    output_tokens.push_back(next_token);
    ++generated;

    // Check for EOS
    if (next_token == config_.eos_token_id)
      break;

    prev_token = next_token;
    ++position;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream_));
  auto decode_end = std::chrono::high_resolution_clock::now();
  stats_.decode_time_ms =
      std::chrono::duration<float, std::milli>(decode_end - decode_start)
          .count();
  stats_.tokens_generated = static_cast<int>(output_tokens.size());

  if (stats_.decode_time_ms > 0) {
    stats_.tokens_per_second =
        stats_.tokens_generated / (stats_.decode_time_ms / 1000.0f);
  }

  // Release sequence
  kv_cache_->releaseSequence(seq_id);

  return output_tokens;
}

void InferenceEngine::prefill(const std::vector<int> &tokens, int seq_id) {
  int num_tokens = static_cast<int>(tokens.size());
  if (num_tokens <= 0) {
    return;
  }

  // Embed tokens
  std::vector<int> h_tokens(tokens);
  DeviceBuffer<int> d_tokens(num_tokens);
  d_tokens.copyFromHost(h_tokens.data(), num_tokens, stream_);

  embedTokens(d_tokens.data(), num_tokens, hidden_states_);

  // Forward through all layers
  for (auto &layer : layers_) {
    layer->forwardPrefill(hidden_states_, *kv_cache_, seq_id, num_tokens,
                          stream_);
  }

  kv_cache_->advanceSeqLen(seq_id, num_tokens);
}

int InferenceEngine::decodeStep(int seq_id, int position, int token_id,
                                const GenerationConfig &config) {
  DeviceBuffer<int> d_token(1);
  d_token.copyFromHost(&token_id, 1, stream_);

  half *token_state = hidden_states_ + position * config_.hidden_dim;
  embedTokens(d_token.data(), 1, token_state);

  // Forward through all layers for single token
  for (auto &layer : layers_) {
    layer->forward(token_state, *kv_cache_, seq_id, position, stream_);
  }

  kv_cache_->advanceSeqLen(seq_id, 1);

  return sampleFromHidden(token_state, config);
}

int InferenceEngine::sampleFromHidden(half *hidden_state,
                                      const GenerationConfig &config) {
  finalNorm(hidden_state, hidden_state, 1);
  computeLogits(hidden_state, 1, logits_);

  CUDA_CHECK(cudaStreamSynchronize(stream_));

  std::vector<half> h_logits(config_.vocab_size);
  CUDA_CHECK(cudaMemcpy(h_logits.data(), logits_,
                        config_.vocab_size * sizeof(half),
                        cudaMemcpyDeviceToHost));

  return sample(h_logits.data(), config);
}

void InferenceEngine::embedTokens(const int *tokens, int num_tokens,
                                  half *output) {
  kernels::gather_embeddings(tokens, weights_.token_embedding, output, num_tokens,
                             config_.hidden_dim, config_.vocab_size, stream_);
}

void InferenceEngine::computeLogits(const half *hidden_states, int num_tokens,
                                    half *logits) {
  // LM head projection: hidden_states @ lm_head.T
  if (weights_.lm_head.isValid()) {
    kernels::w8a16_matmul(hidden_states, weights_.lm_head.data,
                          weights_.lm_head.scales, logits, num_tokens,
                          config_.vocab_size, config_.hidden_dim,
                          weights_.lm_head.group_size, stream_);
  }
}

void InferenceEngine::finalNorm(const half *input, half *output,
                                int num_tokens) {
  if (weights_.final_norm_weight) {
    kernels::rmsnorm(input, weights_.final_norm_weight, output, num_tokens,
                     config_.hidden_dim, config_.rms_norm_eps, stream_);
  }
}

int InferenceEngine::sample(const half *logits,
                            const GenerationConfig &config) {
  if (!config.do_sample) {
    return sampleGreedy(logits, config_.vocab_size);
  }

  if (config.top_p < 1.0f) {
    return sampleTopP(logits, config_.vocab_size, config.top_p,
                      config.temperature);
  }

  if (config.top_k > 0) {
    return sampleTopK(logits, config_.vocab_size, config.top_k,
                      config.temperature);
  }

  return sampleTemperature(logits, config_.vocab_size, config.temperature);
}

// Greedy sampling: return argmax
int InferenceEngine::sampleGreedy(const half *logits, int vocab_size) {
  if (!logits || vocab_size <= 0) {
    return 0;
  }

  int max_idx = 0;
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
int InferenceEngine::sampleTemperature(const half *logits, int vocab_size,
                                       float temperature, unsigned seed) {
  if (!logits || vocab_size <= 0) {
    return 0;
  }
  temperature = std::max(temperature, 1e-5f);

  std::vector<float> probs(vocab_size);
  float max_logit = __half2float(logits[0]);

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

  // Sample from distribution
  std::mt19937 gen(seed ? seed : std::random_device{}());
  std::discrete_distribution<int> dist(probs.begin(), probs.end());

  return dist(gen);
}

// Top-k sampling
int InferenceEngine::sampleTopK(const half *logits, int vocab_size, int k,
                                float temperature, unsigned seed) {
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

  std::partial_sort(
      logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  // Apply temperature and softmax to top-k
  std::vector<float> probs(k);
  float max_logit = logit_pairs[0].first;
  float sum = 0.0f;

  for (int i = 0; i < k; ++i) {
    probs[i] = std::exp((logit_pairs[i].first - max_logit) / temperature);
    sum += probs[i];
  }

  for (int i = 0; i < k; ++i) {
    probs[i] /= sum;
  }

  // Sample
  std::mt19937 gen(seed ? seed : std::random_device{}());
  std::discrete_distribution<int> dist(probs.begin(), probs.end());

  return logit_pairs[dist(gen)].second;
}

// Top-p (nucleus) sampling
int InferenceEngine::sampleTopP(const half *logits, int vocab_size, float p,
                                float temperature, unsigned seed) {
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
  float max_logit = logit_pairs[0].first;
  std::vector<float> probs(vocab_size);
  float sum = 0.0f;

  for (int i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp((logit_pairs[i].first - max_logit) / temperature);
    sum += probs[i];
  }

  // Normalize and find cutoff
  float cumsum = 0.0f;
  int cutoff = vocab_size;
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

  // Sample
  std::mt19937 gen(seed ? seed : std::random_device{}());
  std::discrete_distribution<int> dist(probs.begin(), probs.begin() + cutoff);

  return logit_pairs[dist(gen)].second;
}

} // namespace tiny_llm
