// tiny_llm_bench: 可复现的端到端 benchmark 驱动。
//
// 指标定义（与 DEVELOPMENT_PLAN.md 任务 2.1 一致）：
//   TTFT (ms)   从调用 generate() 到第一个新 token 采样完成的墙钟时间
//               （含 prefill + 第一次 logits；用 max_new_tokens=1 的一次
//               generate 墙钟测量）。
//   TPOT (ms)   decode 阶段墙钟时间 ÷（生成 token 数 − 1）；
//               只生成 1 个 token 时记 N/A。
//   decode tok/s  1 / TPOT × 1000
//   峰值显存 (MB) 加载模型前 cudaMemGetInfo 与 generate 完成后之差。
//
// 统计口径：warmup 次不统计；正式迭代 warmup 之外的 iters 次；
// 输出 mean / p50 / p95 / min / max；迭代间显式 cudaDeviceSynchronize。
//
// 只调用 InferenceEngine 公开 API，不复制采样/生成内部代码。

#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/inference_engine.h"
#include "tiny_llm/tokenizer.h"
#include "w8a16_matmul.cuh" // g_force_reference 诊断开关
#include <algorithm>
#include <chrono>
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
};

struct TimingStats {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
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

void printUsage(const char *program_name) {
    std::cout << "Usage: " << program_name
              << " <model.gguf> --prompt \"...\" --max-tokens 128 --warmup 3 --iters 10 "
                 "[--json] [--use-reference]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --prompt TEXT      Prompt to benchmark (default: 你好)\n";
    std::cout << "  --max-tokens N     Tokens to generate per iteration (default: 128)\n";
    std::cout << "  --warmup N         Warmup iterations, not counted (default: 3)\n";
    std::cout << "  --iters N          Timed iterations (default: 10)\n";
    std::cout << "  --json             Emit one JSON line (machine readable)\n";
    std::cout << "  --use-reference    Force reference w8a16 kernels (diagnostic)\n";
    std::cout << "  -h, --help         Show this help\n";
}

// 单次 generate 的墙钟耗时（内部包含 cudaDeviceSynchronize）。
double timeGenerate(tiny_llm::InferenceEngine *engine, const std::vector<int> &prompt_tokens,
                    int max_new_tokens, int *generated_out = nullptr) {
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
    if (generated_out) {
        *generated_out = static_cast<int>(result.value().size());
    }
    return msBetween(start, end);
}

int runBenchmark(const BenchOptions &opt) {
    // 加载模型前记录一次可用显存，用于峰值显存差值估算。
    size_t free_before = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));

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
    std::vector<double> decode_tok_per_s;
    int                 last_generated = 0;

    const int total_runs = opt.warmup + opt.iters;
    for (int i = 0; i < total_runs; ++i) {
        // TTFT：单 token generate 的墙钟 ≈ prefill + 第一次 logits。
        int gen1 = 0;
        double ttft = timeGenerate(engine.get(), prompt_tokens, 1, &gen1);
        // 主 run：完整 max_tokens 生成；decode 墙钟 ≈ 总墙钟 − TTFT。
        int gen_n = 0;
        double total = timeGenerate(engine.get(), prompt_tokens, opt.max_tokens, &gen_n);
        double decode = total - ttft;
        last_generated = gen_n;

        if (i >= opt.warmup) {
            ttft_ms.push_back(ttft);
            if (gen_n > 1) {
                double tpot = decode / static_cast<double>(gen_n - 1);
                tpot_ms.push_back(tpot);
                decode_tok_per_s.push_back(1000.0 / tpot);
            }
        }
    }

    size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_mem));
    const double peak_mb =
        (free_before > free_after) ? static_cast<double>(free_before - free_after) / (1024.0 * 1024.0)
                                   : 0.0;

    const TimingStats ttft_stats = computeStats(ttft_ms);
    const TimingStats tpot_stats = computeStats(tpot_ms);

    if (opt.json) {
        std::cout << "{"
                  << "\"model\":\"" << opt.model_path << "\","
                  << "\"prompt_tokens\":" << prompt_tokens.size() << ","
                  << "\"new_tokens\":" << last_generated << ","
                  << "\"ttft_ms\":{\"mean\":" << ttft_stats.mean
                  << ",\"p50\":" << ttft_stats.p50 << ",\"p95\":" << ttft_stats.p95
                  << ",\"min\":" << ttft_stats.min << ",\"max\":" << ttft_stats.max << "},"
                  << "\"tpot_ms\":{\"mean\":" << tpot_stats.mean
                  << ",\"p50\":" << tpot_stats.p50 << ",\"p95\":" << tpot_stats.p95
                  << ",\"min\":" << tpot_stats.min << ",\"max\":" << tpot_stats.max << "},"
                  << "\"decode_tok_per_s\":" << (tpot_stats.mean > 0.0 ? 1000.0 / tpot_stats.mean : 0.0)
                  << ",\"peak_mem_mb\":" << peak_mb
                  << "}" << std::endl;
        return 0;
    }

    std::cout << "=== tiny_llm_bench ===" << std::endl;
    std::cout << "Model:          " << opt.model_path << std::endl;
    std::cout << "Prompt tokens:  " << prompt_tokens.size() << std::endl;
    std::cout << "New tokens:     " << last_generated << " (max " << opt.max_tokens << ")"
              << std::endl;
    std::cout << "Warmup / Iters: " << opt.warmup << " / " << opt.iters << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nTTFT (ms)   mean " << std::setw(10) << ttft_stats.mean << "  p50 "
              << std::setw(10) << ttft_stats.p50 << "  p95 " << std::setw(10) << ttft_stats.p95
              << "  min " << std::setw(10) << ttft_stats.min << "  max " << std::setw(10)
              << ttft_stats.max << std::endl;
    if (!tpot_ms.empty()) {
        std::cout << "TPOT (ms)   mean " << std::setw(10) << tpot_stats.mean << "  p50 "
                  << std::setw(10) << tpot_stats.p50 << "  p95 " << std::setw(10)
                  << tpot_stats.p95 << "  min " << std::setw(10) << tpot_stats.min << "  max "
                  << std::setw(10) << tpot_stats.max << std::endl;
        std::cout << "decode tok/s mean " << std::setw(10)
                  << (tpot_stats.mean > 0.0 ? 1000.0 / tpot_stats.mean : 0.0) << std::endl;
    } else {
        std::cout << "TPOT (ms)   N/A (only 1 token generated per iteration)" << std::endl;
    }
    std::cout << "Peak memory delta (MB): " << std::setw(10) << peak_mb << std::endl;

    // 交叉校验：GenerationStats 内部计时（非基准，仅供参考）。
    const auto &stats = engine->getStats();
    std::cout << "\n[cross-check] prefill_time_ms=" << stats.prefill_time_ms
              << " decode_time_ms=" << stats.decode_time_ms << std::endl;
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    BenchOptions opt;
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
            opt.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--warmup") {
            if (i + 1 >= argc) {
                std::cerr << "--warmup requires a value" << std::endl;
                return 1;
            }
            opt.warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters") {
            if (i + 1 >= argc) {
                std::cerr << "--iters requires a value" << std::endl;
                return 1;
            }
            opt.iters = std::stoi(argv[++i]);
        } else if (arg == "--json") {
            opt.json = true;
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

    if (opt.use_reference) {
        tiny_llm::kernels::g_force_reference = true;
        std::cout << "[diagnostic] forcing reference w8a16 kernel" << std::endl;
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
