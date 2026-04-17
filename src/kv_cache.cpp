#include "tiny_llm/kv_cache.h"
#include "tiny_llm/logger.h"
#include "tiny_llm/validator.h"
#include <algorithm>

namespace tiny_llm {

KVCacheManager::KVCacheManager(const KVCacheConfig &config) : config_(config) {

  // Calculate per-slot memory size
  // Each slot stores K and V for all layers
  // K: [num_layers, num_heads, max_seq_len, head_dim]
  // V: [num_layers, num_heads, max_seq_len, head_dim]
  size_t kv_per_layer = static_cast<size_t>(config_.num_heads) *
                        config_.max_seq_len * config_.head_dim;
  size_t kv_total = kv_per_layer * config_.num_layers * 2; // K and V
  slot_size_ = kv_total * sizeof(half);

  // Total pool size for all batch slots
  pool_size_ = slot_size_ * config_.max_batch_size;

  // Allocate GPU memory pool
  CUDA_CHECK(cudaMalloc(&memory_pool_, pool_size_));
  CUDA_CHECK(cudaMemset(memory_pool_, 0, pool_size_));

  // Initialize slots
  slots_.resize(config_.max_batch_size);
  for (auto &slot : slots_) {
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

size_t KVCacheManager::calculateOffset(int slot_idx, int layer_idx,
                                       bool is_value) const {
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
    return Result<int>::err("Invalid max_len: " + std::to_string(max_len) +
                            " (must be 1-" +
                            std::to_string(config_.max_seq_len) + ")");
  }

  // Find free slot
  int slot_idx = findFreeSlot();
  if (slot_idx < 0) {
    return Result<int>::err("KV cache exhausted: no free slots available. "
                            "Used: " +
                            std::to_string(getUsedMemory()) +
                            " bytes, "
                            "Total: " +
                            std::to_string(getTotalMemory()) + " bytes");
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

Result<void> KVCacheManager::releaseSequence(int seq_id) {
  auto it = seq_to_slot_.find(seq_id);
  if (it == seq_to_slot_.end()) {
    TLLM_WARN("releaseSequence: sequence not found: {}", seq_id);
    return Result<void>::err("Sequence not found: " + std::to_string(seq_id));
  }

  int slot_idx = it->second;
  slots_[slot_idx].active = false;
  slots_[slot_idx].seq_id = -1;
  slots_[slot_idx].current_len = 0;

  seq_to_slot_.erase(it);

  TLLM_DEBUG("KVCache: released sequence {} (slot {})", seq_id, slot_idx);
  return Result<void>::ok();
}

Result<std::pair<half *, half *>>
KVCacheManager::getCacheChecked(int seq_id, int layer_idx) {
  auto it = seq_to_slot_.find(seq_id);
  if (it == seq_to_slot_.end()) {
    return Result<std::pair<half *, half *>>::err("Sequence not found: " +
                                                   std::to_string(seq_id));
  }

  auto layer_result =
      Validator::validateLayerIndex(layer_idx, config_.num_layers, "getCache");
  if (layer_result.isErr()) {
    return Result<std::pair<half *, half *>>::err(layer_result.error());
  }

  int slot_idx = it->second;
  size_t k_offset = calculateOffset(slot_idx, layer_idx, false);
  size_t v_offset = calculateOffset(slot_idx, layer_idx, true);

  return Result<std::pair<half *, half *>>::ok(
      {memory_pool_ + k_offset, memory_pool_ + v_offset});
}

std::pair<half *, half *> KVCacheManager::getCache(int seq_id, int layer_idx) {
  auto result = getCacheChecked(seq_id, layer_idx);
  if (result.isErr()) {
    TLLM_WARN("getCache: {}", result.error());
    return {nullptr, nullptr};
  }
  return result.value();
}

Result<void> KVCacheManager::appendKV(int seq_id, int layer_idx, const half *new_k,
                                       const half *new_v, int num_tokens,
                                       cudaStream_t stream) {
  // Validate pointers
  auto ptr_result = Validator::validateNotNull(new_k, "new_k");
  if (ptr_result.isErr()) {
    TLLM_ERROR("appendKV: {}", ptr_result.error());
    return ptr_result;
  }
  ptr_result = Validator::validateNotNull(new_v, "new_v");
  if (ptr_result.isErr()) {
    TLLM_ERROR("appendKV: {}", ptr_result.error());
    return ptr_result;
  }

  // Validate num_tokens
  if (num_tokens <= 0) {
    TLLM_ERROR("appendKV: invalid num_tokens: {}", num_tokens);
    return Result<void>::err("appendKV: num_tokens must be positive: " +
                              std::to_string(num_tokens));
  }

  // Validate layer index
  auto layer_result =
      Validator::validateLayerIndex(layer_idx, config_.num_layers, "appendKV");
  if (layer_result.isErr()) {
    TLLM_ERROR("{}", layer_result.error());
    return layer_result;
  }

  // Find sequence
  auto it = seq_to_slot_.find(seq_id);
  if (it == seq_to_slot_.end()) {
    TLLM_ERROR("appendKV: sequence not found: {}", seq_id);
    return Result<void>::err("appendKV: sequence not found: " +
                              std::to_string(seq_id));
  }

  int slot_idx = it->second;
  auto &slot = slots_[slot_idx];

  // Write position is always current_len — all layers see the same value
  // because advanceSeqLen() is called ONCE after all layers have appended.
  int write_pos = slot.current_len;

  // Check if we have space
  if (write_pos + num_tokens > slot.max_len) {
    TLLM_ERROR("appendKV: cache overflow. seq_id={}, write_pos={}, num_tokens={}, max_len={}",
               seq_id, write_pos, num_tokens, slot.max_len);
    return Result<void>::err(
        "appendKV: cache overflow. Current: " + std::to_string(write_pos) +
        ", appending: " + std::to_string(num_tokens) +
        ", max: " + std::to_string(slot.max_len));
  }

  // Calculate destination offsets
  auto [k_cache, v_cache] = getCache(seq_id, layer_idx);
  if (!k_cache || !v_cache) {
    TLLM_ERROR("appendKV: failed to get cache pointers");
    return Result<void>::err("appendKV: failed to get cache pointers");
  }

  size_t pos_offset =
      static_cast<size_t>(write_pos) * config_.num_heads * config_.head_dim;
  size_t copy_size = static_cast<size_t>(num_tokens) * config_.num_heads *
                     config_.head_dim * sizeof(half);

  CUDA_CHECK(cudaMemcpyAsync(k_cache + pos_offset, new_k, copy_size,
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(v_cache + pos_offset, new_v, copy_size,
                             cudaMemcpyDeviceToDevice, stream));

  TLLM_TRACE("appendKV: seq_id={}, layer={}, num_tokens={}, write_pos={}",
             seq_id, layer_idx, num_tokens, write_pos);
  return Result<void>::ok();
}

Result<void> KVCacheManager::advanceSeqLen(int seq_id, int num_tokens) {
  if (num_tokens <= 0) {
    TLLM_ERROR("advanceSeqLen: invalid num_tokens: {}", num_tokens);
    return Result<void>::err("advanceSeqLen: num_tokens must be positive: " +
                              std::to_string(num_tokens));
  }

  auto it = seq_to_slot_.find(seq_id);
  if (it == seq_to_slot_.end()) {
    TLLM_ERROR("advanceSeqLen: sequence not found: {}", seq_id);
    return Result<void>::err("advanceSeqLen: sequence not found: " +
                              std::to_string(seq_id));
  }

  auto &slot = slots_[it->second];
  int old_len = slot.current_len;
  slot.current_len = std::min(slot.current_len + num_tokens, slot.max_len);

  TLLM_TRACE("advanceSeqLen: seq_id={}, {} -> {}", seq_id, old_len, slot.current_len);
  return Result<void>::ok();
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
  for (const auto &slot : slots_) {
    if (slot.active) {
      active_count++;
    }
  }
  return active_count * slot_size_;
}

size_t KVCacheManager::getTotalMemory() const { return pool_size_; }

size_t KVCacheManager::getFreeMemory() const {
  return getTotalMemory() - getUsedMemory();
}

int KVCacheManager::getActiveSequenceCount() const {
  int count = 0;
  for (const auto &slot : slots_) {
    if (slot.active) {
      count++;
    }
  }
  return count;
}

} // namespace tiny_llm
