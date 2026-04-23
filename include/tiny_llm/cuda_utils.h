#pragma once

#include "tiny_llm/result.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <exception>
#include <sstream>
#include <string>

namespace tiny_llm {

// CUDA error checking macro (throws exception)
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            throw tiny_llm::CudaException(err, __FILE__, __LINE__);                                \
        }                                                                                          \
    } while (0)

// CUDA error checking macro (returns Result<void>)
// Use in functions that return Result<T>
#define CUDA_CHECK_RESULT(call)                                                                    \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            return tiny_llm::Result<void>::err(std::string("CUDA error at ") + __FILE__ + ":" +    \
                                               std::to_string(__LINE__) + ": " +                   \
                                               cudaGetErrorString(err));                           \
        }                                                                                          \
    } while (0)

// CUDA error checking for destructors (logs error, doesn't throw)
// Requires TLLM_ERROR macro from logger.h to be available
#define CUDA_CHECK_DESTROY(call)                                                                   \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            /* Log error but don't throw in destructor */                                          \
            fprintf(stderr, "CUDA error in destructor at %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err));                                                      \
        }                                                                                          \
    } while (0)

// CUDA exception class
class CudaException : public std::exception {
  public:
    CudaException(cudaError_t err, const char *file, int line)
        : error_(err), file_(file), line_(line) {
        std::ostringstream oss;
        oss << "CUDA error " << static_cast<int>(err) << " at " << file << ":" << line << ": "
            << cudaGetErrorString(err);
        message_ = oss.str();
    }

    const char *what() const noexcept override { return message_.c_str(); }

    cudaError_t error() const { return error_; }
    const char *file() const { return file_; }
    int         line() const { return line_; }

  private:
    cudaError_t error_;
    const char *file_;
    int         line_;
    std::string message_;
};

// GPU memory info helper
struct MemoryInfo {
    size_t free;
    size_t total;
    size_t used;
};

inline MemoryInfo getGPUMemoryInfo() {
    MemoryInfo info;
    CUDA_CHECK(cudaMemGetInfo(&info.free, &info.total));
    info.used = info.total - info.free;
    return info;
}

// CUDA stream wrapper
class CudaStream {
  public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

    ~CudaStream() {
        if (stream_) {
            // Use safe check in destructor (no exceptions)
            cudaError_t err = cudaStreamDestroy(stream_);
            if (err != cudaSuccess) {
                // Log error but don't throw in destructor
                fprintf(stderr, "CUDA error in CudaStream destructor: %s\n",
                        cudaGetErrorString(err));
            }
        }
    }

    // Non-copyable
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    // Movable
    CudaStream(CudaStream &&other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }

    CudaStream &operator=(CudaStream &&other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaError_t err = cudaStreamDestroy(stream_);
                if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA error in CudaStream move assignment: %s\n",
                            cudaGetErrorString(err));
                }
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

  private:
    cudaStream_t stream_ = nullptr;
};

// Device memory RAII wrapper
template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_) {
            cudaError_t err = cudaFree(data_);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error in DeviceBuffer destructor: %s\n",
                        cudaGetErrorString(err));
            }
        }
    }

    // Non-copyable
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    // Movable
    DeviceBuffer(DeviceBuffer &&other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            if (data_) {
                cudaError_t err = cudaFree(data_);
                if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA error in DeviceBuffer move assignment: %s\n",
                            cudaGetErrorString(err));
                }
            }
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T       *data() { return data_; }
    const T *data() const { return data_; }
    size_t   size() const { return count_; }
    size_t   bytes() const { return count_ * sizeof(T); }

    void copyFromHost(const T *host_data, size_t count, cudaStream_t stream = 0) {
        CUDA_CHECK(
            cudaMemcpyAsync(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void copyToHost(T *host_data, size_t count, cudaStream_t stream = 0) const {
        CUDA_CHECK(
            cudaMemcpyAsync(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

  private:
    T     *data_ = nullptr;
    size_t count_ = 0;
};

} // namespace tiny_llm
