#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/types.h"
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char **argv) {
  std::cout << "Tiny-LLM Inference Engine" << std::endl;
  std::cout << "=========================" << std::endl;

  // Check CUDA availability
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA devices found!" << std::endl;
    return 1;
  }

  // Print device info
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "Total Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB"
            << std::endl;
  std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;

  // Get memory info
  try {
    auto mem_info = tiny_llm::getGPUMemoryInfo();
    std::cout << "Free Memory: " << mem_info.free / (1024 * 1024) << " MB"
              << std::endl;
  } catch (const tiny_llm::CudaException &e) {
    std::cerr << "CUDA Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\nTiny-LLM ready for inference!" << std::endl;

  return 0;
}
