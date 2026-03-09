#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <unordered_map>
#include "tiny_llm/types.h"
#include "tiny_llm/result.h"
#include "tiny_llm/cuda_utils.h"

namespace tiny_llm {

// Per-sequence cache state
struct SequenceCache {
    int seq_id = -1;
    int current_len = 0;
    int max_len = 0;
    bool active = false;
};

// KV Cache Manager
// Manages GPU memory for key-value caches across all layers and sequences
class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& config);
    ~KVCacheManager();
    
    // Non-copyable
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    
    // Allocate cache for a new sequence
    // Returns sequence ID on success, error on failure
    Result<int> allocateSequence(int max_len);
    
    // Release cache for completed sequence
    void releaseSequence(int seq_id);
    
    // Get cache pointers for a sequence at specific layer
    // Returns (k_cache, v_cache) pointers
    std::pair<half*, half*> getCache(int seq_id, int layer_idx);
    
    // Append new KV pairs to cache (does NOT advance seq_len)
    void appendKV(int seq_id, int layer_idx, 
                  const half* new_k, const half* new_v, 
                  int num_tokens, cudaStream_t stream = 0);
    
    // Advance sequence length after all layers have appended
    // Call ONCE per step, after all layers' appendKV calls
    void advanceSeqLen(int seq_id, int num_tokens);
    
    // Get current sequence length
    int getSeqLen(int seq_id) const;
    
    // Check if sequence exists
    bool hasSequence(int seq_id) const;
    
    // Memory statistics
    size_t getUsedMemory() const;
    size_t getTotalMemory() const;
    size_t getFreeMemory() const;
    
    // Get number of active sequences
    int getActiveSequenceCount() const;
    
    // Get config
    const KVCacheConfig& getConfig() const { return config_; }

private:
    // Calculate memory offset for a specific sequence, layer, and type (K or V)
    size_t calculateOffset(int slot_idx, int layer_idx, bool is_value) const;
    
    // Find free slot
    int findFreeSlot() const;
    
    KVCacheConfig config_;
    
    // GPU memory pool
    half* memory_pool_ = nullptr;
    size_t pool_size_ = 0;
    
    // Slot management
    std::vector<SequenceCache> slots_;
    std::unordered_map<int, int> seq_to_slot_;  // seq_id -> slot_idx
    
    // Next sequence ID
    int next_seq_id_ = 0;
    
    // Per-slot memory size
    size_t slot_size_ = 0;
};

} // namespace tiny_llm
