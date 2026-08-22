#include "paged_kv.cuh"

namespace tiny_llm {
namespace kernels {

namespace {

// 把连续 [num_tokens, chunk_dim] 的 src 按块表散布到 pool。
// 绝对位置 = position + t；块内偏移 = abs % block_size；
// pool 传入的是"本层"指针（调用方负责 layer 偏移）。
__global__ void paged_scatter_blocks_kernel(const half *__restrict__ src,
                                            half *__restrict__ pool,
                                            const int *__restrict__ block_table,
                                            int num_tokens, int position, int block_size,
                                            int chunk_dim, int max_num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * chunk_dim;
    if (idx >= total) return;
    int t = idx / chunk_dim;
    int c = idx - t * chunk_dim;
    int abs = position + t;
    int b = abs / block_size;
    int block_id = block_table[b];
    // 块 id 值域防护：坏块表（调用方 bug/恶意输入）不得越界写 pool
    if (block_id < 0 || block_id >= max_num_blocks) return;
    int within = abs - b * block_size;
    pool[((size_t)block_id * block_size + within) * chunk_dim + c] = src[idx];
}

// 从块表把 [visible_tokens, chunk_dim] 连续读回 dst。
__global__ void paged_gather_blocks_kernel(half *__restrict__ dst,
                                           const half *__restrict__ pool,
                                           const int *__restrict__ block_table,
                                           int visible_tokens, int block_size, int chunk_dim,
                                           int max_num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = visible_tokens * chunk_dim;
    if (idx >= total) return;
    int t = idx / chunk_dim;
    int c = idx - t * chunk_dim;
    int b = t / block_size;
    int block_id = block_table[b];
    // 块 id 值域防护：越界读会触发 illegal address 毒化整个上下文，写 0 兜底
    if (block_id < 0 || block_id >= max_num_blocks) {
        dst[idx] = __float2half(0.0f);
        return;
    }
    int within = t - b * block_size;
    dst[idx] = pool[((size_t)block_id * block_size + within) * chunk_dim + c];
}

} // namespace

void paged_scatter_blocks(const half *src, half *pool, const int *block_table, int num_tokens,
                          int position, int block_size, int chunk_dim, int max_num_blocks,
                          cudaStream_t stream) {
    if (src == nullptr || pool == nullptr || block_table == nullptr || num_tokens <= 0 ||
        block_size <= 0 || chunk_dim <= 0 || max_num_blocks <= 0) {
        return;
    }
    const int total = num_tokens * chunk_dim;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    paged_scatter_blocks_kernel<<<grid, block, 0, stream>>>(src, pool, block_table, num_tokens,
                                                            position, block_size, chunk_dim,
                                                            max_num_blocks);
}

void paged_gather_blocks(half *dst, const half *pool, const int *block_table, int visible_tokens,
                         int block_size, int chunk_dim, int max_num_blocks,
                         cudaStream_t stream) {
    if (dst == nullptr || pool == nullptr || block_table == nullptr || visible_tokens <= 0 ||
        block_size <= 0 || chunk_dim <= 0 || max_num_blocks <= 0) {
        return;
    }
    const int total = visible_tokens * chunk_dim;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    paged_gather_blocks_kernel<<<grid, block, 0, stream>>>(dst, pool, block_table, visible_tokens,
                                                           block_size, chunk_dim,
                                                           max_num_blocks);
}

} // namespace kernels
} // namespace tiny_llm
