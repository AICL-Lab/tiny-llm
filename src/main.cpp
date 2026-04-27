#include "tiny_llm/cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <string>

namespace {

constexpr const char *VERSION = "2.0.2";
constexpr const char *PROJECT_NAME = "Tiny-LLM Inference Engine";

void printVersion() {
    std::cout << PROJECT_NAME << " v" << VERSION << std::endl;
    std::cout << "A lightweight CUDA C++ library for LLM inference with W8A16 quantization"
              << std::endl;
}

void printHelp(const char *program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [MODEL_PATH]" << std::endl;
    std::cout << std::endl;
    std::cout << PROJECT_NAME << " - " << VERSION << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help     Show this help message and exit" << std::endl;
    std::cout << "  -v, --version  Show version information and exit" << std::endl;
    std::cout << "  --info         Show detailed CUDA device information" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  MODEL_PATH     Path to model file (.gguf or binary format)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                    # Show CUDA readiness" << std::endl;
    std::cout << "  " << program_name << " --info            # Show detailed device info"
              << std::endl;
    std::cout << "  " << program_name << " model.gguf        # Load GGUF model (partial support)"
              << std::endl;
}

void printDetailedDeviceInfo() {
    int         device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return;
    }

    std::cout << "=== CUDA Device Information ===" << std::endl;
    std::cout << "Device Count: " << device_count << std::endl;
    std::cout << std::endl;

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get device " << dev
                      << " properties: " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "--- Device " << dev << ": " << prop.name << " ---" << std::endl;
        std::cout << "  Compute Capability:     " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory:    "
                  << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        std::cout << "  Shared Memory per Block:" << prop.sharedMemPerBlock / 1024.0 << " KB"
                  << std::endl;
        std::cout << "  Registers per Block:    " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp Size:              " << prop.warpSize << std::endl;
        std::cout << "  Max Threads per Block:  " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads Dimension:  [" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Dimension:     [" << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << "  SM Count:               " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per SM:     " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Clock Rate:             " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Clock Rate:      " << prop.memoryClockRate / 1000 << " MHz"
                  << std::endl;
        std::cout << "  Memory Bus Width:       " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  L2 Cache Size:          " << prop.l2CacheSize / 1024.0 << " KB"
                  << std::endl;
        std::cout << "  Concurrent Kernels:     " << (prop.concurrentKernels ? "Yes" : "No")
                  << std::endl;
        std::cout << "  Unified Addressing:     " << (prop.unifiedAddressing ? "Yes" : "No")
                  << std::endl;

        // Get memory info
        size_t free_mem = 0, total_mem = 0;
        cudaSetDevice(dev);
        err = cudaMemGetInfo(&free_mem, &total_mem);
        if (err == cudaSuccess) {
            std::cout << "  Free Memory:            " << free_mem / (1024.0 * 1024.0) << " MB"
                      << std::endl;
            std::cout << "  Used Memory:            " << (total_mem - free_mem) / (1024.0 * 1024.0)
                      << " MB" << std::endl;
        }
        std::cout << std::endl;
    }
}

} // namespace

int main(int argc, char **argv) {
    // Parse command line arguments
    bool        show_help = false;
    bool        show_version = false;
    bool        show_info = false;
    std::string model_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg == "-v" || arg == "--version") {
            show_version = true;
        } else if (arg == "--info") {
            show_info = true;
        } else if (arg[0] != '-') {
            model_path = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::cerr << "Use --help for usage information." << std::endl;
            return 1;
        }
    }

    // Handle --help
    if (show_help) {
        printHelp(argv[0]);
        return 0;
    }

    // Handle --version
    if (show_version) {
        printVersion();
        return 0;
    }

    // Basic CUDA check
    std::cout << PROJECT_NAME << std::endl;
    std::cout << "=========================" << std::endl;

    int         device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    // Handle --info
    if (show_info) {
        printDetailedDeviceInfo();
        return 0;
    }

    // Print basic device info
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;

    try {
        auto mem_info = tiny_llm::getGPUMemoryInfo();
        std::cout << "Free Memory: " << mem_info.free / (1024 * 1024) << " MB" << std::endl;
    } catch (const tiny_llm::CudaException &e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return 1;
    }

    // Handle model path argument
    if (!model_path.empty()) {
        if (model_path.size() >= 5 && model_path.substr(model_path.size() - 5) == ".gguf") {
            std::cout << "\nRuntime note: GGUF parsing is partial and runtime GGUF loading is not "
                         "supported yet."
                      << std::endl;
            std::cout << "Use the test binary format consumed by ModelLoader::loadBin() for "
                         "end-to-end loading."
                      << std::endl;
        } else {
            std::cout << "\nRuntime note: the demo binary currently reports CUDA readiness only."
                      << std::endl;
            std::cout << "Model execution still requires integrating a supported load/generate "
                         "path in the demo."
                      << std::endl;
        }
    } else {
        std::cout << "\nThis demo currently reports CUDA readiness only." << std::endl;
        std::cout << "Pass a model path to see supported format notes." << std::endl;
        std::cout << "Use --help for more options." << std::endl;
    }

    return 0;
}
