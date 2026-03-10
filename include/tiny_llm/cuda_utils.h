#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <exception>
#include <sstream>
#include <string>

namespace tiny_llm {

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw tiny_llm::CudaException(err, __FILE__, __LINE__);                  \
    }                                                                          \
  } while (0)

// CUDA exception class
class CudaException : public std::exception {
public:
  CudaException(cudaError_t err, const char *file, int line)
      : error_(err), file_(file), line_(line) {
    std::ostringstream oss;
    oss << "CUDA error " << static_cast<int>(err) << " at " << file << ":"
        << line << ": " << cudaGetErrorString(err);
    message_ = oss.str();
  }

  const char *what() const noexcept override { return message_.c_str(); }

  cudaError_t error() const { return error_; }
  const char *file() const { return file_; }
  int line() const { return line_; }

private:
  cudaError_t error_;
  const char *file_;
  int line_;
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
      cudaStreamDestroy(stream_);
    }
  }

  // Non-copyable
  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  // Movable
  CudaStream(CudaStream &&other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  CudaStream &operator=(CudaStream &&other) noexcept {
    if (this != &other) {
      if (stream_) {
        cudaStreamDestroy(stream_);
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
template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count > 0) {
      CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (data_) {
      cudaFree(data_);
    }
  }

  // Non-copyable
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  // Movable
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (data_) {
        cudaFree(data_);
      }
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  T *data() { return data_; }
  const T *data() const { return data_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

  void copyFromHost(const T *host_data, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync(data_, host_data, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

  void copyToHost(T *host_data, size_t count, cudaStream_t stream = 0) const {
    CUDA_CHECK(cudaMemcpyAsync(host_data, data_, count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
  }

private:
  T *data_ = nullptr;
  size_t count_ = 0;
};

} // namespace tiny_llm
