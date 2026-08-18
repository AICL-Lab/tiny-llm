// tiny_llm_kernel_bench: decode-path kernel microbenchmark.
//
// Purpose: replace ncu / nsys stats (unavailable on this box — WSL2 without
// perf-counter permission and missing nsys importer) with a reproducible,
// in-repo measurement of "how many ms does each kernel take".
//
// Measures the exact shapes used by Qwen2.5-0.5B decode:
//   W8A16 GEMM      M=1, K=896, N in {128, 896, 4864}
//   W8A16 GEMM down M=1, K=4864, N=896
//   FP16 lm_head    M=1, K=896, N=151936
//   attention_decode S in {8,32,64,128}, Hq=14, Hkv=2, D=64
//   rmsnorm         batch=1, hidden=896
//   RoPE            num_tokens=1, Hq=14, Hkv=2, D=64, pos=0
//   add             n=896
//   silu_mul        n=4864
//
// Timing: warmup 20 runs, then 200 runs (lm_head 100), cudaDeviceSynchronize
// before and after the loop, std::chrono::steady_clock mean ms.
// Output: CSV lines <name>,<shape>,<ms> for copy-paste into reports.
//
// This file deliberately contains NO optimization experiments — it only calls
// the existing public kernel interfaces.

#include "attention.cuh"
#include "elementwise.cuh"
#include "rmsnorm.cuh"
#include "rope.cuh"
#include "w8a16_matmul.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const char *msg) {
    std::fprintf(stderr, "kernel_bench: %s (%s)\n", msg, cudaGetErrorString(cudaGetLastError()));
    std::exit(1);
}

void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel_bench: %s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Warm up `warmup` times, then time `iters` runs with a sync before/after.
// Each call to `f` launches work on the default stream.
template <typename F>
double bench(F &&f, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) f();
    check(cudaDeviceSynchronize(), "sync before timed loop");
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) f();
    check(cudaDeviceSynchronize(), "sync after timed loop");
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           static_cast<double>(iters);
}

// ---------------------------------------------------------------------------
// W8A16 GEMM  (weight [K, N], scales [K/group, N])
// ---------------------------------------------------------------------------
double benchW8A16(const char *name, const char *shape, int M, int N, int K, int group_size,
                  int warmup, int iters) {
    const int scale_rows = (K + group_size - 1) / group_size;

    std::vector<__half> h_input(static_cast<size_t>(M) * K, __float2half(0.5f));
    std::vector<int8_t> h_weight(static_cast<size_t>(K) * N, 1);
    std::vector<__half> h_scales(static_cast<size_t>(scale_rows) * N, __float2half(0.5f));
    std::vector<__half> h_output(static_cast<size_t>(M) * N);

    __half   *d_input = nullptr, *d_scales = nullptr, *d_output = nullptr;
    int8_t   *d_weight = nullptr;
    check(cudaMalloc(&d_input, h_input.size() * sizeof(__half)), "cudaMalloc input");
    check(cudaMalloc(&d_weight, h_weight.size()), "cudaMalloc weight");
    check(cudaMalloc(&d_scales, h_scales.size() * sizeof(__half)), "cudaMalloc scales");
    check(cudaMalloc(&d_output, h_output.size() * sizeof(__half)), "cudaMalloc output");
    check(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half),
                     cudaMemcpyHostToDevice), "copy input");
    check(cudaMemcpy(d_weight, h_weight.data(), h_weight.size(), cudaMemcpyHostToDevice),
          "copy weight");
    check(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(__half),
                     cudaMemcpyHostToDevice), "copy scales");

    double ms = bench([&] {
        tiny_llm::kernels::w8a16_matmul(d_input, d_weight, d_scales, d_output, M, N, K,
                                        group_size, 0);
    }, warmup, iters);

    std::printf("w8a16_matmul,%s,%.4f\n", shape, ms);
    check(cudaFree(d_input), "cudaFree input");
    check(cudaFree(d_weight), "cudaFree weight");
    check(cudaFree(d_scales), "cudaFree scales");
    check(cudaFree(d_output), "cudaFree output");
    return ms;
}

// ---------------------------------------------------------------------------
// FP16 lm_head  (weight [K, N])
// ---------------------------------------------------------------------------
double benchFP16(const char *name, const char *shape, int M, int N, int K, int warmup, int iters) {
    std::vector<__half> h_input(static_cast<size_t>(M) * K, __float2half(0.5f));
    std::vector<__half> h_weight(static_cast<size_t>(K) * N, __float2half(0.5f));
    std::vector<__half> h_output(static_cast<size_t>(M) * N);

    __half *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
    check(cudaMalloc(&d_input, h_input.size() * sizeof(__half)), "cudaMalloc input");
    check(cudaMalloc(&d_weight, h_weight.size() * sizeof(__half)), "cudaMalloc weight");
    check(cudaMalloc(&d_output, h_output.size() * sizeof(__half)), "cudaMalloc output");
    check(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__half),
                     cudaMemcpyHostToDevice), "copy input");
    check(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(__half),
                     cudaMemcpyHostToDevice), "copy weight");

    double ms = bench([&] {
        tiny_llm::kernels::fp16_matmul(d_input, d_weight, d_output, M, N, K, 0);
    }, warmup, iters);

    std::printf("fp16_matmul,%s,%.4f\n", shape, ms);
    check(cudaFree(d_input), "cudaFree input");
    check(cudaFree(d_weight), "cudaFree weight");
    check(cudaFree(d_output), "cudaFree output");
    return ms;
}

// ---------------------------------------------------------------------------
// attention_decode  (Q [1, Hq, D], K/V cache [S, Hkv, D], O [1, Hq, D])
// ---------------------------------------------------------------------------
double benchAttentionDecode(int S, int Hq, int Hkv, int D, int warmup, int iters) {
    const size_t q_elems = static_cast<size_t>(Hq) * D;
    const size_t kv_elems = static_cast<size_t>(S) * Hkv * D;

    __half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    int    *d_len = nullptr;
    check(cudaMalloc(&d_q, q_elems * sizeof(__half)), "cudaMalloc q");
    check(cudaMalloc(&d_k, kv_elems * sizeof(__half)), "cudaMalloc k");
    check(cudaMalloc(&d_v, kv_elems * sizeof(__half)), "cudaMalloc v");
    check(cudaMalloc(&d_out, q_elems * sizeof(__half)), "cudaMalloc out");
    check(cudaMalloc(&d_len, sizeof(int)), "cudaMalloc len");
    check(cudaMemset(d_q, 0, q_elems * sizeof(__half)), "memset q");
    check(cudaMemset(d_k, 0, kv_elems * sizeof(__half)), "memset k");
    check(cudaMemset(d_v, 0, kv_elems * sizeof(__half)), "memset v");
    check(cudaMemset(d_out, 0, q_elems * sizeof(__half)), "memset out");
    check(cudaMemcpy(d_len, &S, sizeof(int), cudaMemcpyHostToDevice), "copy len");

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    double ms = bench([&] {
        tiny_llm::kernels::attention_decode(d_q, d_k, d_v, d_out, scale, Hq, Hkv, d_len, D, 0);
    }, warmup, iters);

    char shape[64];
    std::snprintf(shape, sizeof(shape), "S=%d,Hq=%d,Hkv=%d,D=%d", S, Hq, Hkv, D);
    std::printf("attention_decode,%s,%.4f\n", shape, ms);

    check(cudaFree(d_q), "cudaFree q");
    check(cudaFree(d_k), "cudaFree k");
    check(cudaFree(d_v), "cudaFree v");
    check(cudaFree(d_out), "cudaFree out");
    check(cudaFree(d_len), "cudaFree len");
    return ms;
}

// ---------------------------------------------------------------------------
// rmsnorm  (batch=1, hidden)
// ---------------------------------------------------------------------------
double benchRMSNorm(int batch, int hidden, int warmup, int iters) {
    __half *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, static_cast<size_t>(batch) * hidden * sizeof(__half)), "cudaMalloc x");
    check(cudaMalloc(&d_w, static_cast<size_t>(hidden) * sizeof(__half)), "cudaMalloc w");
    check(cudaMalloc(&d_y, static_cast<size_t>(batch) * hidden * sizeof(__half)), "cudaMalloc y");
    check(cudaMemset(d_x, 0, static_cast<size_t>(batch) * hidden * sizeof(__half)), "memset x");
    check(cudaMemset(d_w, 0, static_cast<size_t>(hidden) * sizeof(__half)), "memset w");

    double ms = bench([&] {
        tiny_llm::kernels::rmsnorm(d_x, d_w, d_y, batch, hidden, 1e-6f, 0);
    }, warmup, iters);

    char shape[64];
    std::snprintf(shape, sizeof(shape), "batch=%d,hidden=%d", batch, hidden);
    std::printf("rmsnorm,%s,%.4f\n", shape, ms);

    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_w), "cudaFree w");
    check(cudaFree(d_y), "cudaFree y");
    return ms;
}

// ---------------------------------------------------------------------------
// RoPE (apply_rope_inplace, num_tokens=1)
// ---------------------------------------------------------------------------
double benchRoPE(int Hq, int Hkv, int D, int warmup, int iters) {
    const size_t q_elems = static_cast<size_t>(Hq) * D;
    const size_t k_elems = static_cast<size_t>(Hkv) * D;
    const size_t half_d = static_cast<size_t>(D) / 2;

    __half *d_q = nullptr, *d_k = nullptr;
    float  *d_cos = nullptr, *d_sin = nullptr;
    int    *d_pos = nullptr;
    check(cudaMalloc(&d_q, q_elems * sizeof(__half)), "cudaMalloc q");
    check(cudaMalloc(&d_k, k_elems * sizeof(__half)), "cudaMalloc k");
    check(cudaMalloc(&d_cos, half_d * sizeof(float)), "cudaMalloc cos");
    check(cudaMalloc(&d_sin, half_d * sizeof(float)), "cudaMalloc sin");
    check(cudaMalloc(&d_pos, sizeof(int)), "cudaMalloc pos");
    check(cudaMemset(d_q, 0, q_elems * sizeof(__half)), "memset q");
    check(cudaMemset(d_k, 0, k_elems * sizeof(__half)), "memset k");
    check(cudaMemset(d_cos, 0, half_d * sizeof(float)), "memset cos");
    check(cudaMemset(d_sin, 0, half_d * sizeof(float)), "memset sin");
    const int pos = 0;
    check(cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice), "copy pos");

    double ms = bench([&] {
        tiny_llm::kernels::apply_rope_inplace(d_q, d_k, d_cos, d_sin, 1, d_pos, Hq, Hkv, D, 0);
    }, warmup, iters);

    char shape[64];
    std::snprintf(shape, sizeof(shape), "tokens=1,Hq=%d,Hkv=%d,D=%d,pos=0", Hq, Hkv, D);
    std::printf("apply_rope_inplace,%s,%.4f\n", shape, ms);

    check(cudaFree(d_q), "cudaFree q");
    check(cudaFree(d_k), "cudaFree k");
    check(cudaFree(d_cos), "cudaFree cos");
    check(cudaFree(d_sin), "cudaFree sin");
    check(cudaFree(d_pos), "cudaFree pos");
    return ms;
}

// ---------------------------------------------------------------------------
// add_inplace / silu_mul_inplace
// ---------------------------------------------------------------------------
double benchElementwise(const char *name, int n, int warmup, int iters) {
    __half *d_a = nullptr, *d_b = nullptr;
    check(cudaMalloc(&d_a, static_cast<size_t>(n) * sizeof(__half)), "cudaMalloc a");
    check(cudaMalloc(&d_b, static_cast<size_t>(n) * sizeof(__half)), "cudaMalloc b");
    check(cudaMemset(d_a, 0, static_cast<size_t>(n) * sizeof(__half)), "memset a");
    check(cudaMemset(d_b, 0, static_cast<size_t>(n) * sizeof(__half)), "memset b");

    double ms = 0.0;
    if (std::string(name) == "add_inplace") {
        ms = bench([&] { tiny_llm::kernels::add_inplace(d_a, d_b, n, 0); }, warmup, iters);
    } else if (std::string(name) == "silu_mul_inplace") {
        ms = bench([&] { tiny_llm::kernels::silu_mul_inplace(d_a, d_b, n, 0); }, warmup, iters);
    } else {
        fail("unknown elementwise bench name");
    }

    std::printf("%s,n=%d,%.4f\n", name, n, ms);

    check(cudaFree(d_a), "cudaFree a");
    check(cudaFree(d_b), "cudaFree b");
    return ms;
}

} // namespace

int main() {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::fprintf(stderr, "kernel_bench: no CUDA device found\n");
        return 1;
    }
    check(cudaSetDevice(0), "cudaSetDevice");

    std::printf("name,shape,ms\n");

    // W8A16 GEMM: M=1, K=896, N in {128, 896, 4864}
    benchW8A16("w8a16_matmul", "M=1,K=896,N=128", 1, 128, 896, 128, 20, 200);
    benchW8A16("w8a16_matmul", "M=1,K=896,N=896", 1, 896, 896, 128, 20, 200);
    benchW8A16("w8a16_matmul", "M=1,K=896,N=4864", 1, 4864, 896, 128, 20, 200);
    // W8A16 GEMM down: M=1, K=4864, N=896
    benchW8A16("w8a16_matmul", "M=1,K=4864,N=896", 1, 896, 4864, 128, 20, 200);

    // FP16 lm_head: M=1, K=896, N=151936 (100 iters)
    benchFP16("fp16_matmul", "M=1,K=896,N=151936", 1, 151936, 896, 20, 100);

    // attention_decode: S in {8, 32, 64, 128}, Hq=14, Hkv=2, D=64
    for (int S : {8, 32, 64, 128}) {
        benchAttentionDecode(S, 14, 2, 64, 20, 200);
    }

    // rmsnorm / RoPE / add / silu_mul
    benchRMSNorm(1, 896, 20, 200);
    benchRoPE(14, 2, 64, 20, 200);
    benchElementwise("add_inplace", 896, 20, 200);
    benchElementwise("silu_mul_inplace", 4864, 20, 200);

    return 0;
}
