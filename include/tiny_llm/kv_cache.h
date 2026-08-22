#pragma once

#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/result.h"
#include "tiny_llm/types.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

// Per-sequence cache state
struct SequenceCache {
    int  seq_id = -1;
    int  current_len = 0;
    int  max_len = 0;
    bool active = false;
};

// KV Cache Manager
// Manages GPU memory for key-value caches across all layers and sequences
class KVCacheManager {
  public:
    // Factory method - creates KVCacheManager with Result<T> error handling
    // Use this instead of constructor for proper error handling
    static Result<std::unique_ptr<KVCacheManager>> create(const KVCacheConfig &config);

    ~KVCacheManager();

    // Non-copyable
    KVCacheManager(const KVCacheManager &) = delete;
    KVCacheManager &operator=(const KVCacheManager &) = delete;

    // Non-movable (due to raw pointer management)
    KVCacheManager(KVCacheManager &&) = delete;
    KVCacheManager &operator=(KVCacheManager &&) = delete;

    // Allocate cache for a new sequence
    // Returns sequence ID on success, error on failure
    Result<int> allocateSequence(int max_len);

    // 显式指定 seq_id 分配（C ABI 用：调用方管理序列 id）
    Result<int> allocateSequence(int seq_id, int max_len);

    // Release cache for completed sequence
    // Returns error if sequence not found
    Result<void> releaseSequence(int seq_id);

    // Get cache pointers for a sequence at specific layer
    // Returns (k_cache, v_cache) pointers
    std::pair<half *, half *> getCache(int seq_id, int layer_idx);

    // Get cache pointers with error checking
    // Returns error if sequence not found or invalid layer
    Result<std::pair<half *, half *>> getCacheChecked(int seq_id, int layer_idx);

    // Append new KV pairs to cache (does NOT advance seq_len).
    // The newly appended tokens are written at the current visible tail and may be
    // consumed by the same step's decode attention before advanceSeqLen() is
    // called, but getSeqLen() remains unchanged until the explicit advance.
    // Returns error on invalid input or cache overflow.
    Result<void> appendKV(int seq_id, int layer_idx, const half *new_k, const half *new_v,
                          int num_tokens, cudaStream_t stream = 0);

    // 任务 3.2：device 写位置版本的 appendKV（CUDA Graph 重放前置）。
    // 写位置从 *device_write_pos 读取（host 每次 decode 前更新），而非 host
    // 计算的目标地址 —— 否则 graph 会固化地址、重放时写回同一位置。
    // 要求 *device_write_pos == 该序列当前 visible len（= getSeqLen()）。
    Result<void> appendKV(int seq_id, int layer_idx, const half *new_k, const half *new_v,
                          int num_tokens, const int *device_write_pos, cudaStream_t stream = 0);

    // 设置 append 写位置（device int）。调用方在每次 prefill/decode 前调用。
    void setAppendPos(int pos, cudaStream_t stream = 0);

    // append 写位置（device int）指针，供 appendKV 使用。
    const int *appendPosDevicePtr() const { return append_pos_.data(); }

    // 校验序列剩余容量能否容纳 num_tokens 个新 token（host 侧，CUDA Graph
    // 重放路径不经过 forward 校验时由调用方显式调用）。
    Result<void> validateAppendSpace(int seq_id, int num_tokens) const noexcept;

    // Advance sequence length after all layers have appended.
    // Call exactly ONCE per logical step, after all layers' appendKV calls.
    // Returns error if sequence not found or invalid num_tokens.
    Result<void> advanceSeqLen(int seq_id, int num_tokens);

    // Get current sequence length
    int getSeqLen(int seq_id) const noexcept;

    // Check if sequence exists
    bool hasSequence(int seq_id) const noexcept;

    // Memory statistics。
    // 注意计费口径：getUsedMemory 按"活跃 slot 数 × 单 slot 满额大小"计算
    // （slot 内部按 max_len 预留，不按实际 seq_len 扣减），因此数值是活跃序列
    // 的预留显存而非实际写入字节数；getFreeMemory = 池总量 - 已预留量。
    size_t getUsedMemory() const noexcept;
    size_t getTotalMemory() const noexcept;
    size_t getFreeMemory() const noexcept;

    // Get number of active sequences
    int getActiveSequenceCount() const noexcept;

    // Get config
    const KVCacheConfig &getConfig() const noexcept { return config_; }

  private:
    // Private constructor - use create() factory method
    explicit KVCacheManager(const KVCacheConfig &config) : config_(config) {}

    // Calculate memory offset for a specific sequence, layer, and type (K or V)
    size_t calculateOffset(int slot_idx, int layer_idx, bool is_value) const;

    // Find free slot
    int findFreeSlot() const;

    KVCacheConfig config_;

    // GPU memory pool
    half  *memory_pool_ = nullptr;
    size_t pool_size_ = 0;

    // Slot management
    std::vector<SequenceCache>   slots_;
    std::unordered_map<int, int> seq_to_slot_; // seq_id -> slot_idx

    // Next sequence ID
    int next_seq_id_ = 0;

    // Per-slot memory size
    size_t slot_size_ = 0;

    // 任务 3.2：append 写位置（device int，CUDA Graph 重放前置）
    DeviceBuffer<int> append_pos_;
    // 修复：setAppendPos 的 host 源缓冲（成员变量，稳定地址）。CUDA Graph
    // capture 会固化 H2D memcpy 的 host 指针并在重放时读取，若用栈/临时
    // 变量地址则依赖栈地址复用（未定义行为）。改为成员变量后重放稳定。
    int append_pos_host_ = 0;
};

} // namespace tiny_llm
