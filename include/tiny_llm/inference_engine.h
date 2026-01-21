#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <memory>
#include "tiny_llm/types.h"
#include "tiny_llm/kv_cache.h"
#include "tiny_llm/transformer.h"
#include "tiny_llm/result.h"

namespace tiny_llm {

// Sampling strategies
enum class SamplingStrategy {
    GREEDY,      // argmax
    TEMPERATURE, // temperature scaling + multinomial
    TOP_K,       // top-k filtering
    TOP_P        // nucleus sampling
};

// Inference engine for LLM text generation
class InferenceEngine {
public:
    // Load model from file
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& model_path,
        const ModelConfig& config
    );
    
    // Constructor with pre-loaded weights
    InferenceEngine(const ModelConfig& config, ModelWeights&& weights);
    ~InferenceEngine();
    
    // Non-copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    
    // Generate tokens from prompt
    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& config
    );
    
    // Get generation statistics
    const GenerationStats& getStats() const { return stats_; }
    
    // Reset statistics
    void resetStats() { stats_ = GenerationStats{}; }
    
    // Sampling functions (public for testing)
    static int sampleGreedy(const half* logits, int vocab_size);
    static int sampleTemperature(const half* logits, int vocab_size, float temperature, unsigned seed = 0);
    static int sampleTopK(const half* logits, int vocab_size, int k, float temperature, unsigned seed = 0);
    static int sampleTopP(const half* logits, int vocab_size, float p, float temperature, unsigned seed = 0);

private:
    // Prefill phase: process all prompt tokens
    void prefill(const std::vector<int>& tokens, int seq_id);
    
    // Decode phase: generate one token
    int decodeStep(int seq_id, int position, int token_id, const GenerationConfig& config);

    // Sample from a single hidden state
    int sampleFromHidden(half* hidden_state, const GenerationConfig& config);
    
    // Sample from logits based on config
    int sample(const half* logits, const GenerationConfig& config);
    
    // Embedding lookup
    void embedTokens(const int* tokens, int num_tokens, half* output);
    
    // Compute logits from hidden states
    void computeLogits(const half* hidden_states, int num_tokens, half* logits);
    
    // Apply RMSNorm
    void finalNorm(const half* input, half* output, int num_tokens);
    
    ModelConfig config_;
    ModelWeights weights_;
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    std::unique_ptr<KVCacheManager> kv_cache_;
    
    // Buffers
    half* hidden_states_ = nullptr;
    half* logits_ = nullptr;
    
    // CUDA stream
    cudaStream_t stream_ = 0;
    
    // Statistics
    GenerationStats stats_;
};

} // namespace tiny_llm
