#include "sampling.cuh"

#include <cmath>

namespace tiny_llm {
namespace kernels {

namespace {

constexpr int kArgmaxThreads = 256;

__device__ bool is_better(float value, int index, float best_value, int best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__global__ void greedy_argmax_kernel(const half *logits, int vocab_size, int *tokens) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    logits += static_cast<size_t>(row) * vocab_size;

    // CPU 基线以 logits[0] 初始化；首项为 NaN 时，所有 `>` 比较都为 false，
    // 因而固定返回 0。显式保留这条边界语义，避免 device 归约改变 greedy 结果。
    __shared__ int first_is_nan;
    if (tid == 0) {
        first_is_nan = isnan(__half2float(logits[0])) ? 1 : 0;
    }
    __syncthreads();
    if (first_is_nan != 0) {
        if (tid == 0) tokens[row] = 0;
        return;
    }

    float best_value = -INFINITY;
    int   best_index = vocab_size;
    for (int index = tid; index < vocab_size; index += blockDim.x) {
        const float value = __half2float(logits[index]);
        // CPU 侧 `value > max_value` 会忽略后续 NaN；这里保持相同语义。
        if (!isnan(value) && is_better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }

    __shared__ float values[kArgmaxThreads];
    __shared__ int   indices[kArgmaxThreads];
    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride &&
            is_better(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
            values[tid] = values[tid + stride];
            indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) tokens[row] = indices[0];
}

} // namespace

void greedy_argmax_batch(const half *logits, int batch_size, int vocab_size, int *tokens,
                         cudaStream_t stream) {
    if (logits == nullptr || tokens == nullptr || batch_size <= 0 || vocab_size <= 0) return;
    greedy_argmax_kernel<<<batch_size, kArgmaxThreads, 0, stream>>>(logits, vocab_size, tokens);
}

void greedy_argmax(const half *logits, int vocab_size, int *token, cudaStream_t stream) {
    greedy_argmax_batch(logits, 1, vocab_size, token, stream);
}

} // namespace kernels
} // namespace tiny_llm
