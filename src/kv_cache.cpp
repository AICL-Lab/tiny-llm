#include "tiny_llm/kv_cache.h"
#include <algorithm>

namespace tiny_llm {

KVCacheManager::KVCacheManager(const KVCacheConfig& config)
    : config_(config) {
    
    // Calculate per-slot memory size
    // Each slot stores K and V for all layers
    // K: [num_layers, num_heads, max_seq_len, head_dim]
    // V: [num_layers, num_heads, max_seq_len, head_dim]
    size_t kv_per_layer = static_cast<size_t>(config_.num_heads) * 
                          config_.max_seq_len * config_.head_dim;
    size_t kv_total = kv_per_layer * config_.num_layers * 2;  // K and V
    slot_size_ = kv_total * sizeof(half);
    
    // Total pool size for all batch slots
    pool_size_ = slot_size_ * config_.max_batch_size;
    
    // Allocate GPU memory pool
    CUDA_CHECK(cudaMalloc(&memory_pool_, pool_size_));
    CUDA_CHECK(cudaMemset(memory_pool_, 0, pool_size_));
    
    // Initialize slots
    slots_.resize(config_.max_batch_size);
    for (auto& slot : slots_) {
        slot.active = false;
        slot.seq_id = -1;
        slot.current_len = 0;
        slot.max_len = config_.max_seq_len;
    }
}

KVCacheManager::~KVCacheManager() {
    if (memory_pool_) {
        cudaFree(memory_pool_);
        memory_pool_ = nullptr;
    }
}

size_t KVCacheManager::calculateOffset(int slot_idx, int layer_idx, bool is_value) const {
    // Memory layout per slot:
    // [Layer 0 K][Layer 0 V][Layer 1 K][Layer 1 V]...
    size_t kv_per_layer = static_cast<size_t>(config_.num_heads) * 
                          config_.max_seq_len * config_.head_dim;
    
    size_t slot_offset = slot_idx * slot_size_ / sizeof(half);
    size_t layer_offset = layer_idx * kv_per_layer * 2;
    size_t kv_offset = is_value ? kv_per_layer : 0;
    
    return slot_offset + layer_offset + kv_offset;
}

int KVCacheManager::findFreeSlot() const {
    for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
        if (!slots_[i].active) {
            return i;
        }
    }
    return -1;
}

Result<int> KVCacheManager::allocateSequence(int max_len) {
    // Validate max_len
    if (max_len <= 0 || max_len > config_.max_seq_len) {
        return Result<int>::err(
            "Invalid max_len: " + std::to_string(max_len) + 
            " (must be 1-" + std::to_string(config_.max_seq_len) + ")");
    }
    
    // Find free slot
    int slot_idx = findFreeSlot();
    if (slot_idx < 0) {
        return Result<int>::err(
            "KV cache exhausted: no free slots available. "
            "Used: " + std::to_string(getUsedMemory()) + " bytes, "
            "Total: " + std::to_string(getTotalMemory()) + " bytes");
    }
    
    // Allocate slot
    int seq_id = next_seq_id_++;
    slots_[slot_idx].seq_id = seq_id;
    slots_[slot_idx].current_len = 0;
    slots_[slot_idx].max_len = max_len;
    slots_[slot_idx].active = true;
    
    seq_to_slot_[seq_id] = slot_idx;
    
    return Result<int>::ok(seq_id);
}

void KVCacheManager::releaseSequence(int seq_id) {
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        return;  // Sequence not found, ignore
    }
    
    int slot_idx = it->second;
    slots_[slot_idx].active = false;
    slots_[slot_idx].seq_id = -1;
    slots_[slot_idx].current_len = 0;
    
    seq_to_slot_.erase(it);
}

std::pair<half*, half*> KVCacheManager::getCache(int seq_id, int layer_idx) {
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        return {nullptr, nullptr};
    }
    
    int slot_idx = it->second;
    size_t k_offset = calculateOffset(slot_idx, layer_idx, false);
    size_t v_offset = calculateOffset(slot_idx, layer_idx, true);
    
    return {memory_pool_ + k_offset, memory_pool_ + v_offset};
}

void KVCacheManager::appendKV(int seq_id, int layer_idx,
                               const half* new_k, const half* new_v,
                               int num_tokens, cudaStream_t stream) {
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        return;  // Sequence not found
    }
    
    int slot_idx = it->second;
    auto& slot = slots_[slot_idx];
    
    // Calculate write position based on current length
    // Note: For multi-layer append, layer 0 should be called LAST to update length
    // Or all layers should be called with the same current_len value
    int write_pos;
    if (layer_idx == 0) {
        // Layer 0 uses current position and will update it
        write_pos = slot.current_len;
        
        // Check if we have space
        if (write_pos + num_tokens > slot.max_len) {
            return;  // No space, silently fail
        }
    } else {
        // Other layers: if current_len was already updated by layer 0,
        // we need to write to the position BEFORE the update
        // This assumes layer 0 is called first
        write_pos = slot.current_len - num_tokens;
        if (write_pos < 0) {
            // Layer 0 hasn't been called yet, use current position
            write_pos = slot.current_len;
        }
    }
    
    // Calculate destination offsets
    auto [k_cache, v_cache] = getCache(seq_id, layer_idx);
    
    // Offset to write position
    size_t pos_offset = static_cast<size_t>(write_pos) * 
                        config_.num_heads * config_.head_dim;
    
    // Copy size
    size_t copy_size = static_cast<size_t>(num_tokens) * 
                       config_.num_heads * config_.head_dim * sizeof(half);
    
    // Copy K and V
    CUDA_CHECK(cudaMemcpyAsync(k_cache + pos_offset, new_k, copy_size,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_cache + pos_offset, new_v, copy_size,
                               cudaMemcpyDeviceToDevice, stream));
    
    // Update length only on layer 0
    if (layer_idx == 0) {
        slot.current_len += num_tokens;
    }
}

int KVCacheManager::getSeqLen(int seq_id) const {
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        return 0;
    }
    return slots_[it->second].current_len;
}

bool KVCacheManager::hasSequence(int seq_id) const {
    return seq_to_slot_.find(seq_id) != seq_to_slot_.end();
}

size_t KVCacheManager::getUsedMemory() const {
    int active_count = 0;
    for (const auto& slot : slots_) {
        if (slot.active) {
            active_count++;
        }
    }
    return active_count * slot_size_;
}

size_t KVCacheManager::getTotalMemory() const {
    return pool_size_;
}

size_t KVCacheManager::getFreeMemory() const {
    return getTotalMemory() - getUsedMemory();
}

int KVCacheManager::getActiveSequenceCount() const {
    int count = 0;
    for (const auto& slot : slots_) {
        if (slot.active) {
            count++;
        }
    }
    return count;
}

} // namespace tiny_llm
