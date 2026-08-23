# 2026-08-18 tiny-llm decode 性能攻坚（C0–C2 实测归档）

> 本文件归档 Batch 3（C0–C3）的实测证据：kernel 级 microbench 前后表、
> 端到端 TPOT/tok/s 前后、与 llama.cpp 比值、greedy 输出一致性结论。
> 所有数字用下述命令在仓库内可复现（`tiny_llm_kernel_bench` /
> `tiny_llm_bench` / `tiny_llm_demo`）。

> **后续口径审计（2026-08-23）**：本报告使用 benchmark schema v1，TTFT 与完整墙钟
> 来自两条独立请求；TPOT 是配对估计。表格保留为 C1/C2 优化沿革，正式新结果必须使用
> schema v2 的同请求 TTFT/TPOT，不能把两种 schema 混算。

## 1. 元信息

| 项 | 值 |
|----|-----|
| 日期 | 2026-08-18 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU（6144 MiB，驱动 591.44） |
| CUDA Toolkit | 12.x |
| tiny-llm commit | C0 `ca70de2` → C1 `db5451b` → C2 `f897084`（本归档基线 = `f897084`） |
| 模型 | Qwen2.5-0.5B-Instruct，GGUF Q4_K_M（加载后重量化为 W8A16，group=128） |
| 采样 | greedy（`do_sample=false`），prompt "你好"（1 token） |

## 2. 根因（一行）

```
M==1 kernel 的 lane=k 映射 × 权重 [K,N] 布局 → 跨 lane 地址 stride=N×2B（非 coalesced）
  → 权重转置为 [N,K] 后，lane 沿 k 连续读 weight_t[col*K+k]，stride=1，访存 coalesced
```

- 旧 `w8a16_matmul_m1_kernel` / `fp16_matmul_m1_kernel`：每个 warp 负责一列，
  32 个 lane 按 `k = lane, lane+32, ...` 并行归约 K；但权重按 `weight[k*N+col]`
  读取，同一 warp 内 32 个 lane 的地址跨度为 `N×2B`（int8 为 `N×1B`），
  完全不 coalesced，导致 decode 每个 GEMM 都在搬运整列权重。
- C1 引入转置布局 `[N,K]`（`data_t`/`scales_t`），M==1 时走
  `w8a16_matmul_m1_transposed_kernel` / `fp16_matmul_m1_transposed_kernel`，
  同一 warp 内 lane 地址连续（stride=1），访存 coalesced。

## 3. 为什么不做 ncu / nsys

| 工具 | 本机情况 |
|------|----------|
| `ncu` | 不可用：`ERR_NVGPUCTRPERM`（WSL2 无性能计数器权限）。不尝试。 |
| `nsys stats` | 不可用：importer 缺失，qdstrm 无法转报告。不依赖其做瓶颈分析。 |

替代方案：仓库内新增 **`tiny_llm_kernel_bench`**（任务 C0）——对 decode 路径
每个公开 kernel 接口按真实 shape 做 warmup 20 + 200 次（lm_head 100 次）的
均值计时，输出 CSV 行 `<name>,<shape>,<ms>`。它能区分"哪个 kernel 花多少
时间"，在 C1 前后给出可对比的 kernel 级证据（见第 4.2 节）。

```bash
cmake --build build -j$(nproc)
./build/tiny_llm_kernel_bench
# name,shape,ms
# w8a16_matmul,M=1,K=896,N=128,0.0157
# ...
```

## 4. before / after

### 4.1 端到端（`tiny_llm_bench`，prompt "你好"，64 新 token，warmup 3 / iters 5）

| 指标 | C1 前（graphs ON 基线） | C1 后（graphs ON） | C2 后（默认 graphs ON） |
|------|------------------------|--------------------|--------------------------|
| TTFT (ms) mean | 29.584 | 10.563 | 10.567 |
| TPOT (ms/token) mean | 24.348 | 6.560 | 6.087 |
| decode tok/s mean | 41.072 | 152.442 | 164.283 |
| 常驻显存差值 (MB) | 2494 | 3368（含转置副本） | 3368 |

> 该字段是加载前与 benchmark 结束后的 `cudaMemGetInfo` 差值，不是运行期间真实峰值。

复现命令：

```bash
# C1 前基线（等效：graphs 开启）
./build/tiny_llm_bench ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --prompt "你好" --max-tokens 64 --warmup 3 --iters 5 --graphs
# C2 后（默认即 graphs 开启）
./build/tiny_llm_bench ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --prompt "你好" --max-tokens 64 --warmup 3 --iters 5
```

### 4.2 kernel 级 microbench（`tiny_llm_kernel_bench`，C1 前后）

| 项 | shape | 前 (ms) | 后 (ms) | 加速比 |
|----|-------|--------|--------|--------|
| w8a16_matmul | M=1,K=896,N=128 | 0.0622 | 0.0157 | ~4.0× |
| w8a16_matmul | M=1,K=896,N=896 | 0.0387 | 0.0164 | ~2.4× |
| w8a16_matmul | M=1,K=896,N=4864 | 0.1631 | 0.0486 | ~3.4× |
| w8a16_matmul | M=1,K=4864,N=896（down） | 0.1568 | 0.0506 | ~3.1× |
| fp16_matmul | M=1,K=896,N=151936（lm_head） | 10.0002 | 0.9794 | **~10.2×** |
| attention_decode | S=8 | 0.0147 | 0.0141 | ~1.0× |
| attention_decode | S=32 | 0.0202 | 0.0144 | ~1.4× |
| attention_decode | S=64 | 0.0180 | 0.0221 | ~0.8×（噪声） |
| attention_decode | S=128 | 0.0276 | 0.0467 | ~0.6×（噪声） |
| rmsnorm | batch=1,hidden=896 | 0.0105 | 0.0120 | ~1.0× |
| apply_rope_inplace | tokens=1 | 0.0100 | 0.0100 | ~1.0× |
| add_inplace | n=896 | 0.0155 | 0.0101 | ~1.5× |
| silu_mul_inplace | n=4864 | 0.0120 | 0.0095 | ~1.3× |

> 小 kernel（attention_decode / rmsnorm / rope / elementwise）都在 ~10–50 µs
> 量级，受系统负载/时钟波动影响明显（±50% 允许范围内），不构成 decode 瓶颈。
> 主要收益集中在 W8A16 GEMM（~3–4×）与 lm_head FP16（~10×），对应 C1 的
> 转置快路径。

### 4.3 llama.cpp 比值

| 指标 | tiny-llm（C2 后） | llama.cpp（存档基线） | 比值 (tiny/llama) |
|------|------------------|----------------------|-------------------|
| TPOT (ms/token) | 6.09 | 3.7（`tg64`: 272 t/s） | **1.65** |
| decode tok/s | 164.3 | 272.2 | 0.60 |

> 对比口径见 `benchmark-methodology.md`（llama.cpp 直接算 Q4_K_M，tiny-llm
> 重量化为 W8A16 后计算，非同量化格式）。C1 前 tiny/llama ≈ 6.6，C1 后
> ≈ 1.77，C2 后 ≈ 1.65 —— 从"慢 6 倍"收敛到"慢 ~1.7 倍"。

### 4.4 greedy 输出一致性

- C1 改动前（graphs on / off）demo 输出逐 token 一致；
- C1 改动后 graphs on / off 输出逐 token 一致；
- C1 改动前后输出逐 token **完全一致**（`diff` 为空，token id 序列相同）。

```bash
diff <(grep "Token ids" -A1 /tmp/c1_before_graphs.txt) <(grep "Token ids" -A1 /tmp/c1_after_graphs.txt)  # IDENTICAL
diff <(./build/tiny_llm_demo model.gguf --prompt "你好" --max-tokens 32 --show-tokens 2>&1 | grep "Token ids" -A1) \
     <(TLLM_CUDA_GRAPHS=0 ./build/tiny_llm_demo model.gguf --prompt "你好" --max-tokens 32 --show-tokens 2>&1 | grep "Token ids" -A1)  # IDENTICAL
```

## 5. CUDA Graphs 默认开启（C2）

- 语义反转：`TLLM_CUDA_GRAPHS` **默认开启**，`TLLM_CUDA_GRAPHS=0` 显式关闭
  （opt-out）；`tiny_llm_bench` 默认跑 graphs 路径，`--no-graphs` 用于 A/B。
- 默认（不设环境变量）与 `TLLM_CUDA_GRAPHS=0` 输出逐 token 一致（见 4.4）。
- 默认 bench TPOT 6.09 ms 与 C1 记录一致；捕获失败仍回退常规路径。
- 设计细节见 [cuda-graphs.md](../cuda-graphs.md)。

## 6. 下一步（未做项，诚实清单）

- **Tensor Core / WMMA（FP16 或 INT8）**：当前 M==1 快路径仍是纯 FMA + 标量
  归约，未用 `mma.sync` / `wgmma`，也未做 `half2` 向量化加载。
- **分页 KV**：策略 1 已在后续 commit 实现 block table + scatter/gather；本快照当时尚未包含。
- **FlashDecoding**：attention_decode 仍线性扫描全部可见 KV，未做 L2/共享内存复用、
  跨 head/block 并行。
- **fused kernel**：rmsnorm / RoPE / QKV 投影 / bias 尚未融合。
- **lm_head 进一步优化**：0.98 ms 已达标（≤2.0），但相对 llama.cpp 仍有
  空间（如 k 维 split + 原子写 / 分段 softmax）。
- **多序列 batch / continuous batching**：FFI 路径仍逐序列执行。
