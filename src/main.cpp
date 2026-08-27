#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/inference_engine.h"
#include "tiny_llm/tokenizer.h"
#include "w8a16_matmul.cuh" // 诊断：g_force_reference
#include <cuda_runtime.h>
#include <exception>
#include <iomanip>
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
    std::cout << "  --inspect      Parse a GGUF file and print config/tensor summary" << std::endl;
    std::cout << "  --prompt TEXT  Prompt for GPU end-to-end generation (requires GGUF)"
              << std::endl;
    std::cout << "  --max-tokens N Maximum tokens to generate (default: 64)" << std::endl;
    std::cout << "                 (CPU-only, no GPU required)" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  MODEL_PATH     Path to model file (.gguf or binary format)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                    # Show CUDA readiness" << std::endl;
    std::cout << "  " << program_name << " --info            # Show detailed device info"
              << std::endl;
    std::cout << "  " << program_name
              << " --inspect model.gguf # CPU-only GGUF config/tensor summary" << std::endl;
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
        // clockRate / memoryClockRate 在 CUDA 13 已从 cudaDeviceProp 移除,
        // 改用 cudaDeviceGetAttribute 查询(对 CUDA 10+ 全兼容)
        int clock_rate = 0, mem_clock_rate = 0;
        cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, dev);
        cudaDeviceGetAttribute(&mem_clock_rate, cudaDevAttrMemoryClockRate, dev);
        std::cout << "  Clock Rate:             " << clock_rate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Clock Rate:      " << mem_clock_rate / 1000 << " MHz"
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

const char *ggmlTypeName(tiny_llm::GGMLType type) {
    switch (type) {
    case tiny_llm::GGMLType::F32:
        return "F32";
    case tiny_llm::GGMLType::F16:
        return "F16";
    case tiny_llm::GGMLType::Q4_0:
        return "Q4_0";
    case tiny_llm::GGMLType::Q4_1:
        return "Q4_1";
    case tiny_llm::GGMLType::Q5_0:
        return "Q5_0";
    case tiny_llm::GGMLType::Q5_1:
        return "Q5_1";
    case tiny_llm::GGMLType::Q8_0:
        return "Q8_0";
    case tiny_llm::GGMLType::Q8_1:
        return "Q8_1";
    case tiny_llm::GGMLType::Q2_K:
        return "Q2_K";
    case tiny_llm::GGMLType::Q3_K:
        return "Q3_K";
    case tiny_llm::GGMLType::Q4_K:
        return "Q4_K";
    case tiny_llm::GGMLType::Q5_K:
        return "Q5_K";
    case tiny_llm::GGMLType::Q6_K:
        return "Q6_K";
    case tiny_llm::GGMLType::Q8_K:
        return "Q8_K";
    case tiny_llm::GGMLType::I8:
        return "I8";
    case tiny_llm::GGMLType::I16:
        return "I16";
    case tiny_llm::GGMLType::I32:
        return "I32";
    case tiny_llm::GGMLType::I64:
        return "I64";
    case tiny_llm::GGMLType::F64:
        return "F64";
    default:
        return "UNKNOWN";
    }
}

int inspectGGUF(const std::string &path) {
    tiny_llm::GGUFParser parser(path);
    auto                 parse_result = parser.parse();
    if (parse_result.isErr()) {
        std::cerr << "GGUF parse failed: " << parse_result.error() << std::endl;
        return 1;
    }

    const auto &header = parser.getHeader();
    std::cout << "=== GGUF File: " << path << " ===" << std::endl;
    std::cout << "Version: " << header.version << "  Tensors: " << header.tensor_count
              << "  Metadata KV: " << header.metadata_kv_count << std::endl;

    const auto &metadata = parser.getMetadata();
    if (auto arch = metadata.get<std::string>("general.architecture"); !arch.isErr()) {
        std::cout << "Architecture: " << arch.value() << std::endl;
    }
    if (auto name = metadata.get<std::string>("general.name"); !name.isErr()) {
        std::cout << "Name: " << name.value() << std::endl;
    }

    auto config_result = parser.extractModelConfig();
    if (config_result.isErr()) {
        std::cerr << "Config extraction failed: " << config_result.error() << std::endl;
        return 1;
    }
    const auto &config = config_result.value();
    std::cout << "\n--- Model Config ---" << std::endl;
    std::cout << "hidden_dim:       " << config.hidden_dim << std::endl;
    std::cout << "num_layers:       " << config.num_layers << std::endl;
    std::cout << "num_heads:        " << config.num_heads << std::endl;
    std::cout << "num_kv_heads:     " << config.num_kv_heads << std::endl;
    std::cout << "head_dim:         " << config.head_dim << std::endl;
    std::cout << "intermediate_dim: " << config.intermediate_dim << std::endl;
    std::cout << "vocab_size:       " << config.vocab_size << std::endl;
    std::cout << "max_seq_len:      " << config.max_seq_len << std::endl;
    std::cout << "rope_theta:       " << config.rope_theta << std::endl;
    std::cout << "rms_norm_eps:     " << config.rms_norm_eps << std::endl;

    std::cout << "\n--- Tensors ---" << std::endl;
    std::cout << std::left << std::setw(36) << "name" << std::setw(8) << "type" << std::setw(20)
              << "dims" << "bytes" << std::endl;
    uint64_t total_bytes = 0;
    for (const auto &tensor : parser.getTensors()) {
        std::ostringstream dims;
        for (size_t i = 0; i < tensor.dimensions.size(); ++i) {
            if (i > 0) dims << "x";
            dims << tensor.dimensions[i];
        }
        const size_t bytes = tensor.calculateSize();
        total_bytes += bytes;
        std::cout << std::left << std::setw(36) << tensor.name << std::setw(8)
                  << ggmlTypeName(tensor.type) << std::setw(20) << dims.str() << bytes << std::endl;
    }
    std::cout << "\nTotal tensor data: " << total_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    return 0;
}

// GPU 端到端生成：加载 GGUF 模型，编码 prompt，生成并解码输出。
int runGeneration(const std::string &model_path, const std::string &prompt, int max_tokens,
                  bool show_tokens) {
    try {
        tiny_llm::GGUFParser parser(model_path);
        auto                 parse_result = parser.parse();
        if (parse_result.isErr()) {
            std::cerr << "GGUF parse failed: " << parse_result.error() << std::endl;
            return 1;
        }

        auto config_result = parser.extractModelConfig();
        if (config_result.isErr()) {
            std::cerr << "Config extraction failed: " << config_result.error() << std::endl;
            return 1;
        }
        const auto &config = config_result.value();

        auto td_result = tiny_llm::loadTokenizerData(parser.getMetadata());
        if (td_result.isErr()) {
            std::cerr << "Tokenizer data extraction failed: " << td_result.error() << std::endl;
            return 1;
        }
        auto tokenizer_result = tiny_llm::Tokenizer::build(td_result.value());
        if (tokenizer_result.isErr()) {
            std::cerr << "Tokenizer build failed: " << tokenizer_result.error() << std::endl;
            return 1;
        }
        auto tokenizer = std::move(tokenizer_result.value());

        auto engine_result = tiny_llm::InferenceEngine::load(model_path, config);
        if (engine_result.isErr()) {
            std::cerr << "Model load failed: " << engine_result.error() << std::endl;
            return 1;
        }
        auto engine = std::move(engine_result.value());

        std::vector<int> prompt_tokens = tokenizer.encode(prompt);
        std::cout << "Prompt: \"" << prompt << "\" -> " << prompt_tokens.size() << " tokens"
                  << std::endl;
        if (show_tokens) {
            std::cout << "Prompt tokens: ";
            for (int t : prompt_tokens)
                std::cout << t << " ";
            std::cout << std::endl;
        }

        tiny_llm::GenerationConfig gen_config;
        gen_config.max_new_tokens = max_tokens;
        gen_config.do_sample = false; // greedy，便于与 llama.cpp 做确定性对齐

        auto gen_result = engine->generate(prompt_tokens, gen_config);
        if (gen_result.isErr()) {
            std::cerr << "Generation failed: " << gen_result.error() << std::endl;
            return 1;
        }
        const auto &gen_tokens = gen_result.value();

        std::string text = tokenizer.decode(gen_tokens);
        std::cout << "\n--- Generated text ---" << std::endl;
        std::cout << text << std::endl;

        if (show_tokens) {
            std::cout << "\n--- Token ids ---" << std::endl;
            for (size_t i = 0; i < gen_tokens.size(); ++i) {
                std::cout << i << ":" << gen_tokens[i] << " ";
            }
            std::cout << std::endl;
        }

        const auto &stats = engine->getStats();
        std::cout << "\n--- Stats ---" << std::endl;
        std::cout << "Prompt tokens:    " << stats.prompt_tokens << std::endl;
        std::cout << "Generated tokens: " << stats.tokens_generated << std::endl;
        std::cout << "Prefill time:     " << stats.prefill_time_ms << " ms" << std::endl;
        std::cout << "Decode time:      " << stats.decode_time_ms << " ms" << std::endl;
        if (stats.decode_time_ms > 0.0f) {
            std::cout << "Decode throughput: "
                      << stats.tokens_generated / (stats.decode_time_ms / 1000.0f) << " tok/s"
                      << std::endl;
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace

int main(int argc, char **argv) {
    // Parse command line arguments
    bool        show_help = false;
    bool        show_version = false;
    bool        show_info = false;
    bool        show_inspect = false;
    std::string model_path;
    std::string prompt;
    int         max_tokens = 64;
    bool        show_tokens = false;
    bool        use_reference = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg == "-v" || arg == "--version") {
            show_version = true;
        } else if (arg == "--info") {
            show_info = true;
        } else if (arg == "--inspect") {
            show_inspect = true;
        } else if (arg == "--prompt") {
            if (i + 1 >= argc) {
                std::cerr << "--prompt requires a value" << std::endl;
                return 1;
            }
            prompt = argv[++i];
        } else if (arg == "--max-tokens") {
            if (i + 1 >= argc) {
                std::cerr << "--max-tokens requires a value" << std::endl;
                return 1;
            }
            try {
                max_tokens = std::stoi(argv[++i]);
            } catch (const std::exception &) {
                std::cerr << "--max-tokens requires an integer, got: " << argv[i] << std::endl;
                return 1;
            }
            if (max_tokens <= 0) {
                std::cerr << "--max-tokens must be positive: " << max_tokens << std::endl;
                return 1;
            }
        } else if (arg == "--show-tokens") {
            show_tokens = true;
        } else if (arg == "--use-reference") {
            use_reference = true;
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

    // --inspect 为 CPU-only 路径，不需要 GPU
    if (show_inspect) {
        if (model_path.empty()) {
            std::cerr << "--inspect requires a GGUF file path" << std::endl;
            return 1;
        }
        return inspectGGUF(model_path);
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
            if (!prompt.empty()) {
                if (use_reference) {
                    tiny_llm::kernels::g_force_reference = true;
                    std::cout << "[diagnostic] forcing reference w8a16 kernel" << std::endl;
                }
                return runGeneration(model_path, prompt, max_tokens, show_tokens);
            }
            std::cout << "\nRuntime note: pass --prompt \"...\" to run GPU end-to-end "
                         "generation, or --inspect for a CPU-only GGUF summary."
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
