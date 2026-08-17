# llama.cpp 对比基准方法论

> 本页面定义 tiny-llm 与 llama.cpp 在同一硬件/模型上对比的**可复现**方法。
> 原则：所有数字必须能用本文档中的命令原样复现；任何数字必须同时记录
> 硬件、commit、编译选项、命令（与 DEVELOPMENT_PLAN.md 风险区要求一致）。

## 1. 环境记录模板

每次跑对比前，把以下信息填进结果表（`docs/performance/results/` 下归档）：

### 硬件

| 项 | 值（示例） |
|----|-----------|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 显存 | 6144 MiB |
| 驱动 | 591.44 |
| CUDA Toolkit | 12.0（`nvcc --version` 确认） |
| CPU / 内存 | （可选记录） |

### 软件

| 项 | 值（示例） |
|----|-----------|
| tiny-llm commit | `git rev-parse HEAD` |
| tiny-llm 编译 | Release + `-DBUILD_TESTS=ON`（默认 nvcc 目标架构 `native`） |
| llama.cpp commit | `git -C /path/to/llama.cpp rev-parse HEAD` |
| llama.cpp 编译 | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`（默认 GPU 后端） |
| 对比日期 | YYYY-MM-DD |

## 2. 模型与量化差异（必须声明）

- 模型：`Qwen2.5-0.5B-Instruct`，GGUF `Q4_K_M`。
- **llama.cpp 直接在 Q4_K_M 上计算**（GGML 内核原生支持 K 系列量化）。
- **tiny-llm 的路径是：反量化 → 转置 → 重新量化为 W8A16**（INT8 权重 +
  FP16 scale，group_size=128），加载后以 W8A16 权重推理。

因此两者**并非同一种量化格式**，延迟对比反映的是“原生 Q4_K_M 计算” vs
“W8A16 重量化后计算”两个完整路径的差距，不能解读为纯 GEMM 实现差距。

## 3. 公平性声明（对比前必须满足）

1. 相同 prompt、相同 `max_tokens`（= llama.cpp 的 `-n` 减去 prompt 长度后的
   生成数口径需对齐）、相同 greedy 采样（tiny-llm `do_sample=false`，
   llama.cpp 默认 greedy，`-t 1` 关闭采样并行）。
2. llama.cpp 用 `llama-cli -ngl 99`（全层 GPU）或 `llama-bench`，确保
   不会因 CPU 卸载把数字拖低。
3. 预热与迭代次数保持一致：
   - tiny-llm：`--warmup 3 --iters 10`；
   - llama.cpp：`llama-bench` 的 `-r`（repeat）与 `-n` 对齐，或用
     `llama-cli -n` 连续多次取稳定值。
4. 同一块 GPU、同一时刻附近跑（避免其他进程占用显存/算力）。
5. 峰值显存口径一致：都是“进程启动时与运行完成后 `cudaMemGetInfo`
   可用显存之差”，或统一用 `nvidia-smi` 采样峰值。

## 4. 可复制命令

### tiny-llm

```bash
# 在 tiny-llm 仓库根目录
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

./build/tiny_llm_bench /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --prompt "你好" --max-tokens 64 --warmup 3 --iters 10
# 机器可读：追加 --json
```

### llama.cpp（llama-bench，推荐）

```bash
# 在 llama.cpp 仓库 build 目录
./build/bin/llama-bench \
    -m /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -n 64 \
    -t 1 \
    -r 3
```

`llama-bench` 输出 TTFT（`t_load`/`t_prompt` 相关列）、decode 吞吐
（`tgen` 相关列）与 `pp*` / `tg*` 指标，按需对齐到 tiny-llm 的 TTFT / TPOT。

### llama.cpp（llama-cli，端到端兜底）

```bash
./build/bin/llama-cli \
    -m /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -ngl 99 -t 1 -n 64 \
    -p "你好" --no-display-prompt
```

> 注意：llama-cli 的输出带装饰信息，建议用 `llama-bench` 取数字，
> 用 `llama-cli` 做“同一 prompt 同一 greedy 输出逐 token 一致”的正确性抽查。

## 5. 结果表模板

归档位置：`docs/performance/results/<date>-<gpu>.md`（模板见
`docs/performance/results/TEMPLATE.md`）。**实测快照（2026-08-18，RTX 3060
Laptop，详见 [2026-08-18-rtx3060](results/2026-08-18-rtx3060.md)）**：

| 指标 | tiny-llm | llama.cpp | 比值 (tiny/llama) |
|------|----------|-----------|-------------------|
| TTFT (ms) | 22.9（1-token prompt，含首次 logits） | 4.9（`pp1`，仅 prompt 处理，口径不同） | ~4.7（口径见归档） |
| TPOT (ms/token) | 22.1 | 3.7（`tg64`: 272 t/s） | 6.0 |
| decode tok/s | 45.3 | 272.2 | 0.17 |
| 峰值显存 (MB) | 2490 | 未测（同口径） | — |

每张表下方必须附：第 1、2 节的环境快照 + 实际执行的完整命令 + 原始日志路径。

## 6. 预期差距来源分析（先写假设，实测后回填）

对 0.5B 模型 + decode 场景，预期差距主要来自（按影响排序的假设）：

1. **GEMM 实现差距**：llama.cpp 的 Q4_K_M 内核针对 K 量化格式做了 SIMD
   向量化与分块；tiny-llm 的 W8A16 m1 kernel 是每 warp 一列、32 lane 归约
   K 的简单实现，且 lm_head（FP16，[1, hidden] @ [hidden, 151936]）仍占
   decode 大头。
2. **attention 实现差距**：tiny-llm 的 decode attention 未做 KV 缓存
   L2/共享内存复用与页式布局优化。
3. **运行时/launch 开销**：24 层 × 每层多个 kernel，launch 串行无
   CUDA Graph；llama.cpp 对 decode 有专门的流式优化。
4. **continuous batching 缺失**：tiny-llm 单序列；llama.cpp 即使单序列
   也有更紧凑的调度。

> 实测后：把每一行差距归因到具体 kernel（用任务 2.3 的 nsys/ncu 数据），
> 而不是笼统写“实现差距”。

## 7. 不在对比范围内的事项

- 多请求并发 / batch > 1 吞吐（tiny-llm 无 continuous batching）。
- 采样配置差异（temperature/top-p）——对比固定 greedy。
- 非本机、非同一时段的数据。
