#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// 把连续 [num_tokens, chunk_dim] 的 src 按块表散布到 pool。
// 绝对位置 = position + t；块内偏移 = abs % block_size；
// pool 传入的是"本层"指针（调用方负责 layer 偏移）。
// block_table 是 device 端 int 数组（visible_blocks 个物理块 id）。
// max_num_blocks：块 id 合法值域 [0, max_num_blocks)；越界 id 的元素被跳过
// （scatter 不写 / gather 写 0），防止坏块表造成越界访存毒化 CUDA 上下文。
void paged_scatter_blocks(const half *src, half *pool, const int *block_table, int num_tokens,
                          int position, int block_size, int chunk_dim, int max_num_blocks,
                          cudaStream_t stream = 0);

// 从块表把 [visible_tokens, chunk_dim] 连续读回 dst。
void paged_gather_blocks(half *dst, const half *pool, const int *block_table, int visible_tokens,
                         int block_size, int chunk_dim, int max_num_blocks,
                         cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
