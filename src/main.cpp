#include "tiny_llm/cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <string>

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

  try {
    auto mem_info = tiny_llm::getGPUMemoryInfo();
    std::cout << "Free Memory: " << mem_info.free / (1024 * 1024) << " MB"
              << std::endl;
  } catch (const tiny_llm::CudaException &e) {
    std::cerr << "CUDA Error: " << e.what() << std::endl;
    return 1;
  }

  if (argc > 1) {
    std::string model_path = argv[1];
    if (model_path.size() >= 5 &&
        model_path.substr(model_path.size() - 5) == ".gguf") {
      std::cout << "\nRuntime note: GGUF parsing is partial and runtime GGUF loading is not supported yet." << std::endl;
      std::cout << "Use the test binary format consumed by ModelLoader::loadBin() for end-to-end loading." << std::endl;
    } else {
      std::cout << "\nRuntime note: the demo binary currently reports CUDA readiness only." << std::endl;
      std::cout << "Model execution still requires integrating a supported load/generate path in the demo." << std::endl;
    }
  } else {
    std::cout << "\nThis demo currently reports CUDA readiness only." << std::endl;
    std::cout << "Pass a model path to see supported format notes." << std::endl;
  }

  return 0;
}
