# Tiny-LLM Profiling 指南

> 用 nsys / ncu 分析 `tiny_llm_bench`，产出可归档到
> `docs/performance/results/` 的报告。所有命令可直接复制执行；
> 报告模板见 [results/TEMPLATE.md](./results/TEMPLATE.md)。

## 0. 准备

```bash
# 构建 benchmark（含 Release + CUDA）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# 环境信息（报告里必须记录）
nvidia-smi
nvcc --version
git rev-parse HEAD
```

模型：`/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf`（下文用 `$MODEL` 代替）。

## 1. nsys 时间线（kernel 时间分布）

```bash
mkdir -p reports
nsys profile -o reports/decode \
    ./build/tiny_llm_bench "$MODEL" --prompt "你好" \
    --max-tokens 64 --warmup 1 --iters 3
```

分析输出（`nsys stats` 给 kernel 汇总，按耗时降序）：

```bash
nsys stats --report cuda_gpu_kern_sum reports/decode.nsys-rep
```

**必须记录的表格**（复制到结果报告）：

| kernel | 调用次数 | 平均耗时 (µs) | 总耗时 (ms) | 占比 |
|--------|---------|--------------|------------|------|
| w8a16_matmul_m1_kernel | | | | |
| fp16_matmul_m1_kernel | | | | |
| attention_decode_kernel | | | | |
| attention_prefill_kernel | | | | |
| rmsnorm_kernel | | | | |
| rope_kernel | | | | |
| elementwise（含其余） | | | | |
| memcpy D2D / D2H | | | | |

按类别汇总（matmul / attention / rmsnorm / rope / elementwise / memcpy），
用于判断瓶颈类别。

## 2. ncu 单 kernel 分析（以 attention_decode 为例）

```bash
# --set full 全指标；--launch-count 限制次数，--launch-skip 跳过 prefill
ncu --set full --launch-count 3 --launch-skip 10 \
    ./build/tiny_llm_bench "$MODEL" --prompt "你好" --max-tokens 64 --warmup 1 --iters 1
```

针对特定 kernel：

```bash
ncu -k "regex:w8a16_matmul_m1" --set full \
    ./build/tiny_llm_bench "$MODEL" --prompt "你好" --max-tokens 64 --warmup 1 --iters 1
```

**必须记录的指标**：SM 利用率 / 内存吞吐（% of peak）/ 寄存器数 /
占用率 / 是否受内存带宽限制。

## 3. GEMM 形状与 grid/block 清单

decode 阶段每个 GEMM 的 M/N/K 与 launch 配置。可从
`nsys stats` 的 kernel 参数列或源码确认：

| GEMM | M | N | K | kernel | grid | block |
|------|---|---|---|--------|------|-------|
| QKV proj（各层） | 1 | 3×hidden | hidden | w8a16_matmul_m1 | | |
| attention out proj | 1 | hidden | hidden | w8a16_matmul_m1 | | |
| MLP gate/up | 1 | intermediate | hidden | w8a16_matmul_m1 | | |
| MLP down | 1 | hidden | intermediate | w8a16_matmul_m1 | | |
| lm_head（FP16） | 1 | vocab(151936) | hidden | fp16_matmul_m1 | | |

## 4. decode launch 开销（CUDA Graphs 的 before/after）

统计一次 decode step（单 token）的 kernel launch 数量与总 launch 开销：

```bash
# 从 nsys 时间线数出：一次 decode 的 launch 次数
nsys stats --report cuda_gpu_trace reports/decode.nsys-rep \
    | awk 'NR>1 {n++} END {print n " launches total"}'
```

预期：24 层 × 每层约 10+ kernel（rmsnorm×2 + qkv + attention + out_proj +
gate/up/down + rope + appendKV memcpy 等）≈ 250+ 次 launch / decode step。
把这些数字记录为 CUDA Graphs 优化前后的对比基准（阶段 3）。

## 5. 常见瓶颈速查

| 症状 | 结论 | 对策 |
|------|------|------|
| decode 大量小 kernel、GPU 有间隙 | launch 开销受限 | CUDA Graphs、kernel 融合 |
| attention kernel 内存吞吐高 | 带宽受限 | KV 复用、减少读放大 |
| lm_head kernel 占比大 | 大 N GEMM 未优化 | 分块、Tensor Core |
| CPU 时间线有等待 GPU 空隙 | 同步/拷贝未异步化 | 减少 D2H 同步点 |

## 6. 归档

按 [results/TEMPLATE.md](./results/TEMPLATE.md) 生成报告，原始 `.nsys-rep` /
`.ncu-rep` 放仓库外（git-lfs 或本地），仓库内只提交 markdown 摘要与 CSV。
