#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/result.h"
#include <gtest/gtest.h>

using namespace tiny_llm;

// Test Result<T> class
class ResultTest : public ::testing::Test {};

TEST_F(ResultTest, OkResultHoldsValue) {
  auto result = Result<int>::ok(42);
  EXPECT_TRUE(result.isOk());
  EXPECT_FALSE(result.isErr());
  EXPECT_EQ(result.value(), 42);
}

TEST_F(ResultTest, ErrResultHoldsMessage) {
  auto result = Result<int>::err("Something went wrong");
  EXPECT_FALSE(result.isOk());
  EXPECT_TRUE(result.isErr());
  EXPECT_EQ(result.error(), "Something went wrong");
}

TEST_F(ResultTest, ValueOrReturnsValueOnOk) {
  auto result = Result<int>::ok(42);
  EXPECT_EQ(result.valueOr(0), 42);
}

TEST_F(ResultTest, ValueOrReturnsDefaultOnErr) {
  auto result = Result<int>::err("error");
  EXPECT_EQ(result.valueOr(99), 99);
}

TEST_F(ResultTest, AccessingValueOnErrThrows) {
  auto result = Result<int>::err("error");
  EXPECT_THROW(result.value(), std::runtime_error);
}

TEST_F(ResultTest, AccessingErrorOnOkThrows) {
  auto result = Result<int>::ok(42);
  EXPECT_THROW(result.error(), std::runtime_error);
}

TEST_F(ResultTest, MapTransformsValue) {
  auto result = Result<int>::ok(10);
  auto mapped = result.map([](int x) { return x * 2; });
  EXPECT_TRUE(mapped.isOk());
  EXPECT_EQ(mapped.value(), 20);
}

TEST_F(ResultTest, MapPreservesError) {
  auto result = Result<int>::err("error");
  auto mapped = result.map([](int x) { return x * 2; });
  EXPECT_TRUE(mapped.isErr());
  EXPECT_EQ(mapped.error(), "error");
}

TEST_F(ResultTest, VoidResultOk) {
  auto result = Result<void>::ok();
  EXPECT_TRUE(result.isOk());
  EXPECT_FALSE(result.isErr());
}

TEST_F(ResultTest, VoidResultErr) {
  auto result = Result<void>::err("void error");
  EXPECT_FALSE(result.isOk());
  EXPECT_TRUE(result.isErr());
  EXPECT_EQ(result.error(), "void error");
}

TEST_F(ResultTest, StringResult) {
  auto result = Result<std::string>::ok("hello");
  EXPECT_TRUE(result.isOk());
  EXPECT_EQ(result.value(), "hello");
}

// Test CudaException class
class CudaExceptionTest : public ::testing::Test {};

TEST_F(CudaExceptionTest, ExceptionContainsErrorInfo) {
  CudaException ex(cudaErrorMemoryAllocation, "test.cu", 42);

  EXPECT_EQ(ex.error(), cudaErrorMemoryAllocation);
  EXPECT_STREQ(ex.file(), "test.cu");
  EXPECT_EQ(ex.line(), 42);

  std::string what_str = ex.what();
  EXPECT_TRUE(what_str.find("test.cu") != std::string::npos);
  EXPECT_TRUE(what_str.find("42") != std::string::npos);
}

// Test CUDA_CHECK macro (requires CUDA device)
class CudaCheckTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }
};

TEST_F(CudaCheckTest, SuccessfulCallDoesNotThrow) {
  EXPECT_NO_THROW({ CUDA_CHECK(cudaSetDevice(0)); });
}

TEST_F(CudaCheckTest, FailedCallThrowsCudaException) {
  // Try to allocate an impossibly large amount of memory
  void *ptr = nullptr;
  EXPECT_THROW({ CUDA_CHECK(cudaMalloc(&ptr, SIZE_MAX)); }, CudaException);
}

// Test DeviceBuffer
class DeviceBufferTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }
};

TEST_F(DeviceBufferTest, DefaultConstructorCreatesEmptyBuffer) {
  DeviceBuffer<float> buffer;
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.bytes(), 0);
}

TEST_F(DeviceBufferTest, AllocatesMemory) {
  DeviceBuffer<float> buffer(100);
  EXPECT_NE(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size(), 100);
  EXPECT_EQ(buffer.bytes(), 100 * sizeof(float));
}

TEST_F(DeviceBufferTest, MoveConstructor) {
  DeviceBuffer<float> buffer1(100);
  float *ptr = buffer1.data();

  DeviceBuffer<float> buffer2(std::move(buffer1));
  EXPECT_EQ(buffer2.data(), ptr);
  EXPECT_EQ(buffer2.size(), 100);
  EXPECT_EQ(buffer1.data(), nullptr);
  EXPECT_EQ(buffer1.size(), 0);
}

TEST_F(DeviceBufferTest, MoveAssignment) {
  DeviceBuffer<float> buffer1(100);
  DeviceBuffer<float> buffer2(50);
  float *ptr = buffer1.data();

  buffer2 = std::move(buffer1);
  EXPECT_EQ(buffer2.data(), ptr);
  EXPECT_EQ(buffer2.size(), 100);
}

TEST_F(DeviceBufferTest, CopyFromAndToHost) {
  const size_t count = 100;
  std::vector<float> host_data(count);
  for (size_t i = 0; i < count; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  DeviceBuffer<float> buffer(count);
  buffer.copyFromHost(host_data.data(), count);

  std::vector<float> result(count);
  buffer.copyToHost(result.data(), count);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }
}

// Test CudaStream
class CudaStreamTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }
};

TEST_F(CudaStreamTest, CreatesValidStream) {
  CudaStream stream;
  EXPECT_NE(stream.get(), nullptr);
}

TEST_F(CudaStreamTest, MoveConstructor) {
  CudaStream stream1;
  cudaStream_t ptr = stream1.get();

  CudaStream stream2(std::move(stream1));
  EXPECT_EQ(stream2.get(), ptr);
}

TEST_F(CudaStreamTest, Synchronize) {
  CudaStream stream;
  EXPECT_NO_THROW(stream.synchronize());
}

// Test getGPUMemoryInfo
class MemoryInfoTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }
};

TEST_F(MemoryInfoTest, ReturnsValidInfo) {
  auto info = getGPUMemoryInfo();
  EXPECT_GT(info.total, 0);
  EXPECT_GE(info.free, 0);
  EXPECT_EQ(info.used, info.total - info.free);
}
