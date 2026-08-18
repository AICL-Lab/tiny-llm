#pragma once

#include "tiny_llm/kv_cache.h"
#include "tiny_llm/result.h"
#include "tiny_llm/transformer.h"
#include "rope.cuh"  // TLLM-003
#include "tiny_llm/types.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

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
    static Result<std::unique_ptr<InferenceEngine>> load(const std::string &model_path,
                                                         const ModelConfig &config);

    // Constructor with pre-loaded weights
    InferenceEngine(const ModelConfig &config, ModelWeights &&weights);
    ~InferenceEngine();

    // Non-copyable
    InferenceEngine(const InferenceEngine &) = delete;
    InferenceEngine &operator=(const InferenceEngine &) = delete;

    // Generate tokens from prompt
    // Returns Result<vector<int>> with generated tokens or error
    Result<std::vector<int>> generate(const std::vector<int> &prompt_tokens,
                                      const GenerationConfig &config);

    // Get generation statistics
    const GenerationStats &getStats() const { return stats_; }

    // Reset statistics
    void resetStats() { stats_ = GenerationStats{}; }

    // 当前活跃 KV 序列数（任务 4.1 失败路径测试用：超长输入拒绝后不应增加）
    int getActiveSequenceCount() const {
        return kv_cache_ ? kv_cache_->getActiveSequenceCount() : 0;
    }

    // Sampling functions (public for testing)
    static int sampleGreedy(const half *logits, int vocab_size);
    static int sampleTemperature(const half *logits, int vocab_size, float temperature,
                                 unsigned seed = 0);
    static int sampleTopK(const half *logits, int vocab_size, int k, float temperature,
                          unsigned seed = 0);
    static int sampleTopP(const half *logits, int vocab_size, float p, float temperature,
                          unsigned seed = 0);

  private:
    // Prefill phase: process all prompt tokens
    Result<void> prefill(const std::vector<int> &tokens, int seq_id);

    // Decode phase: generate one token
    Result<int> decodeStep(int seq_id, int position, int token_id, const GenerationConfig &config);

    // Sample from a single hidden state
    int sampleFromHidden(half *hidden_state, const GenerationConfig &config);

    // Sample from logits based on config
    int sample(const half *logits, const GenerationConfig &config);

    // Embedding lookup
    void embedTokens(const int *tokens, int num_tokens, half *output);

    // Compute logits from hidden states
    void computeLogits(const half *hidden_states, int num_tokens, half *logits);

    // Apply RMSNorm
    void finalNorm(const half *input, half *output, int num_tokens);

    ModelConfig                                    config_;
    ModelWeights                                   weights_;
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    std::unique_ptr<KVCacheManager>                kv_cache_;

    // 共享中间激活工作区（所有层复用，RAII 自动释放）
    LayerWorkspace workspace_;

    // Buffers
    half *hidden_states_ = nullptr;
    half *logits_ = nullptr;

    // 任务 3.1：decode 可见 KV 长度（device 端 int，长度 1）。
    // attention_decode 从该缓冲读取 visible_len，使 decode kernel 参数不含
    // 每次变化的值 —— CUDA Graph 捕获/重放的前置条件。每次 decodeStep 前
    // 在 stream_ 上 cudaMemcpyAsync 更新。
    DeviceBuffer<int> decode_len_;

    // 任务 3.2（CUDA Graphs）：decode 固定输入缓冲与 RoPE 起始位置。
    // graph_token_: 单 token id（embed 的输入指针必须跨重放稳定）。
    // rope_pos_: RoPE 起始绝对位置（device int，graph 重放前置）。
    DeviceBuffer<int> graph_token_;
    DeviceBuffer<int> rope_pos_;

    // 任务 3.2 / C2：CUDA Graph 状态。默认开启 decode graph 捕获/重放；
    // 环境变量 TLLM_CUDA_GRAPHS=0 显式关闭（opt-out）。捕获失败自动回退
    // 并置 graphs_enabled=false。
    bool          cuda_graphs_enabled_ = false;
    bool          graph_captured_ = false;
    cudaGraph_t   decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;

    // 任务 3.2：decode 的 device 序列（embed + 24 层 forward + finalNorm +
    // computeLogits），graph 捕获/直接执行共用同一函数保证路径一致。
    Result<void> runDecodeDevicePath(half *token_state, int seq_id, int position);

    // 任务 3.2：从 logits_ 采样（sync + D2H + sample，graph 不覆盖）。
    int sampleFromLogits(const GenerationConfig &config);

    // TLLM-003: RoPE cos/sin half cache (FP32)
    float *rope_cos_ = nullptr;
    float *rope_sin_ = nullptr;

    // CUDA stream
    cudaStream_t stream_ = 0;

    // Statistics
    GenerationStats stats_;
};

} // namespace tiny_llm
