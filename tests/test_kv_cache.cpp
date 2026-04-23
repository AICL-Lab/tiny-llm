#include "tiny_llm/kv_cache.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <rapidcheck.h>
// NOTE: rapidcheck/gtest disabled due to GCC 11 std_function bug
// #include <rapidcheck/gtest.h>

using namespace tiny_llm;

class KVCacheTest : public ::testing::Test {
  protected:
    void SetUp() override {
        int         device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        cudaSetDevice(0);
    }

    void TearDown() override { cudaDeviceSynchronize(); }

    KVCacheConfig createConfig(int num_layers = 2, int num_heads = 4, int head_dim = 64,
                               int max_seq_len = 128, int max_batch_size = 4) {
        KVCacheConfig config;
        config.num_layers = num_layers;
        config.num_heads = num_heads;
        config.head_dim = head_dim;
        config.max_seq_len = max_seq_len;
        config.max_batch_size = max_batch_size;
        return config;
    }
};

// Unit tests
TEST_F(KVCacheTest, InitializationAllocatesMemory) {
    auto           config = createConfig();
    KVCacheManager cache(config);

    EXPECT_GT(cache.getTotalMemory(), 0);
    EXPECT_EQ(cache.getUsedMemory(), 0);
    EXPECT_EQ(cache.getActiveSequenceCount(), 0);
}

TEST_F(KVCacheTest, AllocateSequenceReturnsValidId) {
    auto           config = createConfig();
    KVCacheManager cache(config);

    auto result = cache.allocateSequence(64);
    ASSERT_TRUE(result.isOk());

    int seq_id = result.value();
    EXPECT_GE(seq_id, 0);
    EXPECT_TRUE(cache.hasSequence(seq_id));
    EXPECT_EQ(cache.getSeqLen(seq_id), 0);
}

TEST_F(KVCacheTest, AllocateMultipleSequences) {
    auto           config = createConfig(2, 4, 64, 128, 4);
    KVCacheManager cache(config);

    std::vector<int> seq_ids;
    for (int i = 0; i < 4; ++i) {
        auto result = cache.allocateSequence(64);
        ASSERT_TRUE(result.isOk());
        seq_ids.push_back(result.value());
    }

    EXPECT_EQ(cache.getActiveSequenceCount(), 4);

    // All IDs should be unique
    std::sort(seq_ids.begin(), seq_ids.end());
    auto last = std::unique(seq_ids.begin(), seq_ids.end());
    EXPECT_EQ(last, seq_ids.end());
}

TEST_F(KVCacheTest, ExhaustionReturnsError) {
    auto           config = createConfig(2, 4, 64, 128, 2); // Only 2 slots
    KVCacheManager cache(config);

    // Allocate all slots
    ASSERT_TRUE(cache.allocateSequence(64).isOk());
    ASSERT_TRUE(cache.allocateSequence(64).isOk());

    // Third allocation should fail
    auto result = cache.allocateSequence(64);
    EXPECT_TRUE(result.isErr());
    EXPECT_TRUE(result.error().find("exhausted") != std::string::npos);
}

TEST_F(KVCacheTest, ReleaseSequenceFreesSlot) {
    auto           config = createConfig(2, 4, 64, 128, 2);
    KVCacheManager cache(config);

    auto r1 = cache.allocateSequence(64);
    auto r2 = cache.allocateSequence(64);
    ASSERT_TRUE(r1.isOk());
    ASSERT_TRUE(r2.isOk());

    int seq1 = r1.value();

    // Release first sequence
    cache.releaseSequence(seq1);
    EXPECT_FALSE(cache.hasSequence(seq1));
    EXPECT_EQ(cache.getActiveSequenceCount(), 1);

    // Should be able to allocate again
    auto r3 = cache.allocateSequence(64);
    EXPECT_TRUE(r3.isOk());
}

TEST_F(KVCacheTest, GetCacheReturnsValidPointers) {
    auto           config = createConfig();
    KVCacheManager cache(config);

    auto result = cache.allocateSequence(64);
    ASSERT_TRUE(result.isOk());
    int seq_id = result.value();

    for (int layer = 0; layer < config.num_layers; ++layer) {
        auto [k_cache, v_cache] = cache.getCache(seq_id, layer);
        EXPECT_NE(k_cache, nullptr);
        EXPECT_NE(v_cache, nullptr);
        EXPECT_NE(k_cache, v_cache); // K and V should be different
    }
}

TEST_F(KVCacheTest, GetCacheForInvalidSequenceReturnsNull) {
    auto           config = createConfig();
    KVCacheManager cache(config);

    auto [k_cache, v_cache] = cache.getCache(999, 0);
    EXPECT_EQ(k_cache, nullptr);
    EXPECT_EQ(v_cache, nullptr);
}

TEST_F(KVCacheTest, InvalidMaxLenReturnsError) {
    auto           config = createConfig(2, 4, 64, 128, 4);
    KVCacheManager cache(config);

    // Zero length
    auto r1 = cache.allocateSequence(0);
    EXPECT_TRUE(r1.isErr());

    // Negative length
    auto r2 = cache.allocateSequence(-1);
    EXPECT_TRUE(r2.isErr());

    // Exceeds max
    auto r3 = cache.allocateSequence(256);
    EXPECT_TRUE(r3.isErr());
}

TEST_F(KVCacheTest, MemoryAccountingIsCorrect) {
    auto           config = createConfig(2, 4, 64, 128, 4);
    KVCacheManager cache(config);

    size_t initial_free = cache.getFreeMemory();
    size_t total = cache.getTotalMemory();

    EXPECT_EQ(initial_free, total);
    EXPECT_EQ(cache.getUsedMemory(), 0);

    // Allocate one sequence
    auto r1 = cache.allocateSequence(64);
    ASSERT_TRUE(r1.isOk());

    size_t used_after_one = cache.getUsedMemory();
    EXPECT_GT(used_after_one, 0);
    EXPECT_EQ(cache.getFreeMemory(), total - used_after_one);

    // Allocate another
    auto r2 = cache.allocateSequence(64);
    ASSERT_TRUE(r2.isOk());

    size_t used_after_two = cache.getUsedMemory();
    EXPECT_EQ(used_after_two, used_after_one * 2);
}

TEST_F(KVCacheTest, AppendDoesNotAdvanceVisibleSequenceLength) {
    auto           config = createConfig(2, 4, 8, 32, 2);
    KVCacheManager cache(config);

    auto result = cache.allocateSequence(16);
    ASSERT_TRUE(result.isOk());
    int seq_id = result.value();

    int    num_tokens = 3;
    size_t elements = static_cast<size_t>(num_tokens) * config.num_heads * config.head_dim;
    std::vector<half> host_k(elements, __float2half(1.0f));
    std::vector<half> host_v(elements, __float2half(2.0f));
    half             *device_k = nullptr;
    half             *device_v = nullptr;
    cudaMalloc(&device_k, elements * sizeof(half));
    cudaMalloc(&device_v, elements * sizeof(half));
    cudaMemcpy(device_k, host_k.data(), elements * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, host_v.data(), elements * sizeof(half), cudaMemcpyHostToDevice);

    cache.appendKV(seq_id, 0, device_k, device_v, num_tokens);
    cudaDeviceSynchronize();
    EXPECT_EQ(cache.getSeqLen(seq_id), 0);

    cache.advanceSeqLen(seq_id, num_tokens);
    EXPECT_EQ(cache.getSeqLen(seq_id), num_tokens);

    cudaFree(device_k);
    cudaFree(device_v);
}

TEST_F(KVCacheTest, InvalidLayerReturnsNullCachePointers) {
    auto           config = createConfig(2, 4, 8, 32, 2);
    KVCacheManager cache(config);

    auto result = cache.allocateSequence(16);
    ASSERT_TRUE(result.isOk());
    int seq_id = result.value();

    auto [k_cache, v_cache] = cache.getCache(seq_id, -1);
    EXPECT_EQ(k_cache, nullptr);
    EXPECT_EQ(v_cache, nullptr);

    auto [k_cache2, v_cache2] = cache.getCache(seq_id, config.num_layers);
    EXPECT_EQ(k_cache2, nullptr);
    EXPECT_EQ(v_cache2, nullptr);
}

TEST_F(KVCacheTest, InvalidAppendInputsDoNotAdvanceSequenceLength) {
    auto           config = createConfig(2, 4, 8, 32, 2);
    KVCacheManager cache(config);

    auto result = cache.allocateSequence(16);
    ASSERT_TRUE(result.isOk());
    int seq_id = result.value();

    cache.appendKV(seq_id, -1, nullptr, nullptr, 0);
    cache.appendKV(seq_id, config.num_layers, nullptr, nullptr, 1);
    cache.appendKV(seq_id, 0, nullptr, nullptr, 1);
    cudaDeviceSynchronize();

    EXPECT_EQ(cache.getSeqLen(seq_id), 0);
}

TEST_F(KVCacheTest, AdvanceSeqLenClampsToSequenceCapacity) {
    auto           config = createConfig(2, 4, 8, 32, 2);
    KVCacheManager cache(config);

    auto result = cache.allocateSequence(5);
    ASSERT_TRUE(result.isOk());
    int seq_id = result.value();

    cache.advanceSeqLen(seq_id, 100);
    EXPECT_EQ(cache.getSeqLen(seq_id), 5);
}

TEST_F(KVCacheTest, AdvanceSeqLenIgnoresNonPositiveValues) {
    auto           config = createConfig(2, 4, 8, 32, 2);
    KVCacheManager cache(config);

    auto result = cache.allocateSequence(5);
    ASSERT_TRUE(result.isOk());
    int seq_id = result.value();

    cache.advanceSeqLen(seq_id, 0);
    cache.advanceSeqLen(seq_id, -3);
    EXPECT_EQ(cache.getSeqLen(seq_id), 0);
}

// Property-based tests
// Feature: tiny-llm-inference-engine, Property 2: KV Cache Invariants
// Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6
// NOTE: Property-based tests are temporarily disabled due to GCC 11/12
// compatibility issues with rapidcheck's GTest integration.

#if 0
class KVCachePropertyTest : public KVCacheTest {};

RC_GTEST_FIXTURE_PROP(KVCachePropertyTest, AllocationInvariant,
                      (int num_allocs_raw)) {
  int num_allocs = 1 + (std::abs(num_allocs_raw) % 8);

  auto config = createConfig(2, 4, 64, 128, 8);
  KVCacheManager cache(config);

  std::vector<int> allocated_ids;

  for (int i = 0; i < num_allocs; ++i) {
    auto result = cache.allocateSequence(64);

    if (i < config.max_batch_size) {
      // Should succeed
      RC_ASSERT(result.isOk());
      int seq_id = result.value();

      // Invariant: allocated sequence should exist
      RC_ASSERT(cache.hasSequence(seq_id));

      // Invariant: initial length should be 0
      RC_ASSERT(cache.getSeqLen(seq_id) == 0);

      // Invariant: cache pointers should be valid
      for (int layer = 0; layer < config.num_layers; ++layer) {
        auto [k, v] = cache.getCache(seq_id, layer);
        RC_ASSERT(k != nullptr);
        RC_ASSERT(v != nullptr);
      }

      allocated_ids.push_back(seq_id);
    } else {
      // Should fail (exhausted)
      RC_ASSERT(result.isErr());
    }
  }

  // Invariant: active count matches allocations
  RC_ASSERT(cache.getActiveSequenceCount() ==
            static_cast<int>(allocated_ids.size()));
}

RC_GTEST_FIXTURE_PROP(KVCachePropertyTest, ReleaseInvariant,
                      (std::vector<bool> release_pattern)) {
  RC_PRE(!release_pattern.empty());

  int num_seqs = std::min(static_cast<int>(release_pattern.size()), 4);

  auto config = createConfig(2, 4, 64, 128, 4);
  KVCacheManager cache(config);

  // Allocate sequences
  std::vector<int> seq_ids;
  for (int i = 0; i < num_seqs; ++i) {
    auto result = cache.allocateSequence(64);
    RC_ASSERT(result.isOk());
    seq_ids.push_back(result.value());
  }

  size_t used_before = cache.getUsedMemory();
  int active_before = cache.getActiveSequenceCount();

  // Release some sequences based on pattern
  int released_count = 0;
  for (int i = 0; i < num_seqs; ++i) {
    if (release_pattern[i % release_pattern.size()]) {
      cache.releaseSequence(seq_ids[i]);

      // Invariant: released sequence should not exist
      RC_ASSERT(!cache.hasSequence(seq_ids[i]));

      released_count++;
    }
  }

  // Invariant: active count decreased by released count
  RC_ASSERT(cache.getActiveSequenceCount() == active_before - released_count);

  // Invariant: used memory decreased proportionally
  if (released_count > 0) {
    RC_ASSERT(cache.getUsedMemory() < used_before);
  }
}

RC_GTEST_FIXTURE_PROP(KVCachePropertyTest, MemoryInvariant, (int num_ops_raw)) {
  int num_ops = 1 + (std::abs(num_ops_raw) % 20);

  auto config = createConfig(2, 4, 64, 128, 4);
  KVCacheManager cache(config);

  size_t total = cache.getTotalMemory();

  for (int i = 0; i < num_ops; ++i) {
    // Invariant: used + free = total
    RC_ASSERT(cache.getUsedMemory() + cache.getFreeMemory() == total);

    // Invariant: used memory = active_count * slot_size
    // (approximately, since slot_size is internal)

    // Random operation
    if (cache.getActiveSequenceCount() < config.max_batch_size &&
        (i % 2 == 0)) {
      cache.allocateSequence(64);
    } else if (cache.getActiveSequenceCount() > 0) {
      // Release a random sequence (simplified: release first found)
      for (int j = 0; j < 100; ++j) {
        if (cache.hasSequence(j)) {
          cache.releaseSequence(j);
          break;
        }
      }
    }
  }

  // Final invariant check
  RC_ASSERT(cache.getUsedMemory() + cache.getFreeMemory() == total);
}

RC_GTEST_FIXTURE_PROP(KVCachePropertyTest, ExhaustionInvariant, ()) {
  auto config = createConfig(2, 4, 64, 128, 3); // Only 3 slots
  KVCacheManager cache(config);

  // Allocate all slots
  for (int i = 0; i < config.max_batch_size; ++i) {
    auto result = cache.allocateSequence(64);
    RC_ASSERT(result.isOk());
  }

  // Invariant: further allocations should fail
  auto result = cache.allocateSequence(64);
  RC_ASSERT(result.isErr());

  // Invariant: no partial allocation occurred
  RC_ASSERT(cache.getActiveSequenceCount() == config.max_batch_size);

  // Release one and try again
  cache.releaseSequence(0);

  // Should succeed now
  auto result2 = cache.allocateSequence(64);
  RC_ASSERT(result2.isOk());
}
#endif
