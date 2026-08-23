// tiny_llm_bench: 可复现的端到端 benchmark 驱动。
//
// 指标定义（与 DEVELOPMENT_PLAN.md 任务 2.1 一致）：
//   TTFT (ms)   同一次 generate() 从入口到第一个新 token 采样完成的墙钟时间
//               （含校验、KV 分配、prefill 与第一次 logits）。
//   TPOT (ms)   同一次 generate() 的（总墙钟 − TTFT）÷（生成 token 数 − 1）；
//               只生成 1 个 token 时记 N/A。
//   decode tok/s  1 / TPOT × 1000
//   常驻显存差值 (MB) 加载模型前与 benchmark 完成后的 cudaMemGetInfo 差值；
//                   不是进程运行期间的真实峰值。
//
// 统计口径：warmup 次不统计；正式迭代 warmup 之外的 iters 次；
// 输出 mean / p50 / p95 / min / max；迭代间显式 cudaDeviceSynchronize。
//
// 只调用 InferenceEngine 公开 API，不复制采样/生成内部代码。

#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/inference_engine.h"
#include "tiny_llm/logger.h"
#include "tiny_llm/tokenizer.h"
#include "w8a16_matmul.cuh" // g_force_reference 诊断开关
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchOptions {
    std::string model_path;
    std::string prompt = "你好";
    int         max_tokens = 128;
    int         warmup = 3;
    int         iters = 10;
    bool        json = false;
    bool        use_reference = false;
    bool        graphs = false;
    bool        no_graphs = false;
};

struct TimingStats {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct RunMeasurement {
    double                    total_ms = 0.0;
    int                       generated_tokens = 0;
    tiny_llm::GenerationStats stats;
};

TimingStats computeStats(const std::vector<double> &samples) {
    TimingStats s;
    if (samples.empty()) return s;
    std::vector<double> v = samples;
    std::sort(v.begin(), v.end());
    auto quantile = [&](double q) {
        double idx = q * static_cast<double>(v.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = std::min(lo + 1, v.size() - 1);
        double frac = idx - static_cast<double>(lo);
        return v[lo] + (v[hi] - v[lo]) * frac;
    };
    s.min = v.front();
    s.max = v.back();
    s.p50 = quantile(0.50);
    s.p95 = quantile(0.95);
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    return s;
}

double msBetween(const Clock::time_point &a, const Clock::time_point &b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string jsonEscape(const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += ch;
            break;
        }
    }
    return escaped;
}

void printUsage(const char *program_name) {
    std::cout << "Usage: " << program_name
              << " <model.gguf> --prompt \"...\" --max-tokens 128 --warmup 3 --iters 10 "
                 "[--json] [--use-reference]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --prompt TEXT      Prompt to benchmark (default: 你好)\n";
    std::cout << "  --max-tokens N     Tokens to generate per iteration (default: 128)\n";
    std::cout << "  --warmup N         Warmup iterations, not counted (default: 3)\n";
    std::cout << "  --iters N          Timed iterations (default: 10)\n";
    std::cout << "  --json             Emit exactly one JSON object on stdout\n";
    std::cout << "  --use-reference    Force reference w8a16 kernels (diagnostic)\n";
    std::cout << "  --graphs           CUDA Graphs decode is ON by default; flag kept for\n";
    std::cout << "                     compatibility and prints the current state\n";
    std::cout
        << "  --no-graphs        Explicitly disable CUDA Graphs decode (TLLM_CUDA_GRAPHS=0)\n";
    std::cout << "  -h, --help         Show this help\n";
}

// 单次 generate 的墙钟与同请求内部统计（内部包含 cudaDeviceSynchronize）。
RunMeasurement measureGenerate(tiny_llm::InferenceEngine *engine,
                               const std::vector<int> &prompt_tokens, int max_new_tokens) {
    tiny_llm::GenerationConfig gen_config;
    gen_config.max_new_tokens = max_new_tokens;
    gen_config.do_sample = false; // greedy，便于与 llama.cpp 做确定性对齐

    auto start = Clock::now();
    auto result = engine->generate(prompt_tokens, gen_config);
    cudaDeviceSynchronize();
    auto end = Clock::now();

    if (result.isErr()) {
        std::cerr << "generate() failed: " << result.error() << std::endl;
        std::exit(1);
    }
    RunMeasurement measurement;
    measurement.total_ms = msBetween(start, end);
    measurement.generated_tokens = static_cast<int>(result.value().size());
    measurement.stats = engine->getStats();
    return measurement;
}

int runBenchmark(const BenchOptions &opt) {
    // 加载模型前记录一次可用显存，用于常驻显存差值估算。
    size_t free_before = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));

    int            device = 0;
    cudaDeviceProp device_prop{};
    int            driver_version = 0;
    int            runtime_version = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    tiny_llm::GGUFParser parser(opt.model_path);
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

    auto engine_result = tiny_llm::InferenceEngine::load(opt.model_path, config);
    if (engine_result.isErr()) {
        std::cerr << "Model load failed: " << engine_result.error() << std::endl;
        return 1;
    }
    auto engine = std::move(engine_result.value());

    std::vector<int> prompt_tokens = tokenizer.encode(opt.prompt);
    if (prompt_tokens.empty()) {
        std::cerr << "Prompt encoded to zero tokens" << std::endl;
        return 1;
    }

    // ---- 采样 ----
    std::vector<double> ttft_ms;
    std::vector<double> tpot_ms;
    int                 last_generated = 0;

    const int total_runs = opt.warmup + opt.iters;

    // 修复：首次 decodeStep 会执行一次性 CUDA Graph capture（direct execute +
    // 同步 + capture 记录 + EndCapture），开销可达数 ms。若落在测量内（如
    // --warmup 0 时第 0 轮 TTFT），启动开销会被计入 TTFT、decode = total −
    // TTFT 被低估，TPOT 系统性偏低。采样循环前显式预热一次，保证所有
    // 正式测量都走在 graph 重放路径上。
    measureGenerate(engine.get(), prompt_tokens, opt.max_tokens);

    for (int i = 0; i < total_runs; ++i) {
        const RunMeasurement run = measureGenerate(engine.get(), prompt_tokens, opt.max_tokens);
        const double         ttft = run.stats.time_to_first_token_ms;
        const double         decode = run.total_ms - ttft;
        last_generated = run.generated_tokens;

        if (ttft <= 0.0 || decode < 0.0) {
            std::cerr << "Invalid timing sample: total_ms=" << run.total_ms << " ttft_ms=" << ttft
                      << std::endl;
            return 1;
        }

        if (i >= opt.warmup) {
            ttft_ms.push_back(ttft);
            if (run.generated_tokens > 1) {
                double tpot = decode / static_cast<double>(run.generated_tokens - 1);
                tpot_ms.push_back(tpot);
            }
        }
    }

    size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_mem));
    const double resident_delta_mb =
        (free_before > free_after)
            ? static_cast<double>(free_before - free_after) / (1024.0 * 1024.0)
            : 0.0;

    const TimingStats ttft_stats = computeStats(ttft_ms);
    const TimingStats tpot_stats = computeStats(tpot_ms);
    const bool        has_tpot = !tpot_ms.empty();

    if (opt.json) {
        std::cout << "{"
                  << "\"schema_version\":2,"
                  << "\"model\":\"" << jsonEscape(opt.model_path) << "\","
                  << "\"gpu\":{\"name\":\"" << jsonEscape(device_prop.name)
                  << "\",\"total_memory_mib\":" << device_prop.totalGlobalMem / (1024 * 1024)
                  << ",\"cuda_driver_version\":" << driver_version
                  << ",\"cuda_runtime_version\":" << runtime_version << "},"
                  << "\"benchmark\":{\"prompt_tokens\":" << prompt_tokens.size()
                  << ",\"max_tokens\":" << opt.max_tokens << ",\"warmup\":" << opt.warmup
                  << ",\"iterations\":" << opt.iters << "},"
                  << "\"cuda_graphs\":{\"enabled\":"
                  << (engine->cudaGraphsEnabled() ? "true" : "false")
                  << ",\"captured\":" << (engine->cudaGraphCaptured() ? "true" : "false") << "},"
                  << "\"generated_tokens\":" << last_generated << ","
                  << "\"ttft_ms\":{\"mean\":" << ttft_stats.mean << ",\"p50\":" << ttft_stats.p50
                  << ",\"p95\":" << ttft_stats.p95 << ",\"min\":" << ttft_stats.min
                  << ",\"max\":" << ttft_stats.max << "},";
        if (has_tpot) {
            std::cout << "\"tpot_ms\":{\"mean\":" << tpot_stats.mean
                      << ",\"p50\":" << tpot_stats.p50 << ",\"p95\":" << tpot_stats.p95
                      << ",\"min\":" << tpot_stats.min << ",\"max\":" << tpot_stats.max
                      << "},\"decode_tok_per_s\":" << 1000.0 / tpot_stats.mean << ",";
        } else {
            std::cout << "\"tpot_ms\":null,\"decode_tok_per_s\":null,";
        }
        std::cout << "\"resident_memory_delta_mb\":" << resident_delta_mb << "}" << std::endl;
        return 0;
    }

    std::cout << "=== tiny_llm_bench ===" << std::endl;
    std::cout << "Model:          " << opt.model_path << std::endl;
    std::cout << "GPU:            " << device_prop.name << " ("
              << device_prop.totalGlobalMem / (1024 * 1024) << " MiB)" << std::endl;
    std::cout << "Prompt tokens:  " << prompt_tokens.size() << std::endl;
    std::cout << "New tokens:     " << last_generated << " (max " << opt.max_tokens << ")"
              << std::endl;
    std::cout << "Warmup / Iters: " << opt.warmup << " / " << opt.iters << std::endl;
    std::cout << "CUDA Graphs:    " << (engine->cudaGraphsEnabled() ? "enabled" : "disabled")
              << " / " << (engine->cudaGraphCaptured() ? "captured" : "not captured") << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nTTFT (ms)   mean " << std::setw(10) << ttft_stats.mean << "  p50 "
              << std::setw(10) << ttft_stats.p50 << "  p95 " << std::setw(10) << ttft_stats.p95
              << "  min " << std::setw(10) << ttft_stats.min << "  max " << std::setw(10)
              << ttft_stats.max << std::endl;
    if (!tpot_ms.empty()) {
        std::cout << "TPOT (ms)   mean " << std::setw(10) << tpot_stats.mean << "  p50 "
                  << std::setw(10) << tpot_stats.p50 << "  p95 " << std::setw(10) << tpot_stats.p95
                  << "  min " << std::setw(10) << tpot_stats.min << "  max " << std::setw(10)
                  << tpot_stats.max << std::endl;
        std::cout << "decode tok/s mean " << std::setw(10)
                  << (tpot_stats.mean > 0.0 ? 1000.0 / tpot_stats.mean : 0.0) << std::endl;
    } else {
        std::cout << "TPOT (ms)   N/A (only 1 token generated per iteration)" << std::endl;
    }
    std::cout << "Resident memory delta (MB): " << std::setw(10) << resident_delta_mb
              << " (not peak)" << std::endl;

    // 交叉校验：GenerationStats 内部计时（非基准，仅供参考）。
    const auto &stats = engine->getStats();
    std::cout << "\n[cross-check] prefill_time_ms=" << stats.prefill_time_ms
              << " decode_time_ms=" << stats.decode_time_ms << std::endl;
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    BenchOptions opt;
    // 修复：std::stoi 对非数字输入抛 std::invalid_argument、超范围抛
    // std::out_of_range，未捕获会直接 abort。统一在此解析并校验。
    auto parsePositive = [](const char *raw, const char *opt_name, int minimum, int *out) -> bool {
        try {
            *out = std::stoi(raw);
        } catch (const std::exception &) {
            std::cerr << opt_name << " requires an integer, got: " << raw << std::endl;
            return false;
        }
        if (*out < minimum) {
            std::cerr << opt_name << " must be >= " << minimum << ": " << *out << std::endl;
            return false;
        }
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--prompt") {
            if (i + 1 >= argc) {
                std::cerr << "--prompt requires a value" << std::endl;
                return 1;
            }
            opt.prompt = argv[++i];
        } else if (arg == "--max-tokens") {
            if (i + 1 >= argc) {
                std::cerr << "--max-tokens requires a value" << std::endl;
                return 1;
            }
            if (!parsePositive(argv[++i], "--max-tokens", 1, &opt.max_tokens)) return 1;
        } else if (arg == "--warmup") {
            if (i + 1 >= argc) {
                std::cerr << "--warmup requires a value" << std::endl;
                return 1;
            }
            if (!parsePositive(argv[++i], "--warmup", 0, &opt.warmup)) return 1;
        } else if (arg == "--iters") {
            if (i + 1 >= argc) {
                std::cerr << "--iters requires a value" << std::endl;
                return 1;
            }
            if (!parsePositive(argv[++i], "--iters", 1, &opt.iters)) return 1;
        } else if (arg == "--json") {
            opt.json = true;
        } else if (arg == "--graphs") {
            opt.graphs = true;
        } else if (arg == "--no-graphs") {
            opt.no_graphs = true;
        } else if (arg == "--use-reference") {
            opt.use_reference = true;
        } else if (!arg.empty() && arg[0] != '-') {
            opt.model_path = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (opt.graphs && opt.no_graphs) {
        std::cerr << "--graphs and --no-graphs are mutually exclusive" << std::endl;
        return 1;
    }

    if (opt.json) {
        tiny_llm::Logger::init(tiny_llm::LogLevel::OFF);
    }

    if (opt.use_reference) {
        tiny_llm::kernels::g_force_reference = true;
        std::cerr << "[diagnostic] forcing reference w8a16 kernel" << std::endl;
    }

    // 任务 C2：CUDA Graphs 默认开启。--no-graphs 显式关闭（引擎在构造时读
    // 环境变量，须在 InferenceEngine::load 前设置）；--graphs 仅作兼容/诊断，
    // 输出当前状态（默认已开启）。
    if (opt.no_graphs) {
        setenv("TLLM_CUDA_GRAPHS", "0", 1);
        std::cerr << "[diagnostic] disabling CUDA Graphs decode (TLLM_CUDA_GRAPHS=0)" << std::endl;
    } else if (opt.graphs) {
        setenv("TLLM_CUDA_GRAPHS", "1", 1);
        std::cerr << "[diagnostic] enabling CUDA Graphs decode (TLLM_CUDA_GRAPHS=1)" << std::endl;
    }

    if (opt.model_path.empty()) {
        std::cerr << "Error: missing model path." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    int         device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    return runBenchmark(opt);
}
