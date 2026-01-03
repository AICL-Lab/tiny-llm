#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "tiny_llm/cuda_utils.h"

namespace tiny_llm {

// CUDA Stream pool for overlapping computation and memory transfers
class StreamPool {
public:
    explicit StreamPool(int num_streams = 4) : num_streams_(num_streams) {
        streams_.resize(num_streams);
        for (int i = 0; i < num_streams; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        }
    }
    
    ~StreamPool() {
        for (auto& stream : streams_) {
            if (stream) {
                cudaStreamDestroy(stream);
            }
        }
    }
    
    // Non-copyable
    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;
    
    // Move constructible
    StreamPool(StreamPool&& other) noexcept 
        : streams_(std::move(other.streams_)), 
          num_streams_(other.num_streams_),
          current_idx_(other.current_idx_) {
        other.num_streams_ = 0;
        other.current_idx_ = 0;
    }
    
    // Get next stream in round-robin fashion
    cudaStream_t getStream() {
        cudaStream_t stream = streams_[current_idx_];
        current_idx_ = (current_idx_ + 1) % num_streams_;
        return stream;
    }
    
    // Get specific stream
    cudaStream_t getStream(int idx) const {
        return streams_[idx % num_streams_];
    }
    
    // Synchronize all streams
    void synchronizeAll() {
        for (auto& stream : streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    
    // Get number of streams
    int numStreams() const { return num_streams_; }

private:
    std::vector<cudaStream_t> streams_;
    int num_streams_;
    int current_idx_ = 0;
};

// CUDA Event for timing and synchronization
class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }
    
    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }
    
    // Non-copyable
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    // Move constructible
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    
    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }
    
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }
    
    // Elapsed time in milliseconds between two events
    static float elapsedMs(const CudaEvent& start, const CudaEvent& end) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
        return ms;
    }
    
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// Performance configuration for kernel launches
struct KernelConfig {
    int block_size = 256;
    int max_blocks = 65535;
    int shared_mem_size = 0;
    bool use_cooperative_groups = false;
    
    // Auto-tune block size based on kernel requirements
    static KernelConfig autoTune(int elements, int min_threads_per_block = 32) {
        KernelConfig config;
        
        // Choose block size based on element count
        if (elements <= 256) {
            config.block_size = 64;
        } else if (elements <= 1024) {
            config.block_size = 128;
        } else if (elements <= 4096) {
            config.block_size = 256;
        } else {
            config.block_size = 512;
        }
        
        // Ensure minimum threads
        config.block_size = std::max(config.block_size, min_threads_per_block);
        
        return config;
    }
};

// Memory coalescing helper
inline size_t alignTo(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Optimal alignment for GPU memory (128 bytes for coalesced access)
constexpr size_t GPU_MEMORY_ALIGNMENT = 128;

// Aligned memory allocation
inline void* allocateAligned(size_t size) {
    void* ptr = nullptr;
    size_t aligned_size = alignTo(size, GPU_MEMORY_ALIGNMENT);
    CUDA_CHECK(cudaMalloc(&ptr, aligned_size));
    return ptr;
}

} // namespace tiny_llm
