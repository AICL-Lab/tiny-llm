# 2026-08-23 CUDA Graph current-worktree A/B（RTX 3060 Laptop）

> 本报告验证 benchmark schema v2 与 CUDA Graph 方向性收益。工作区包含未提交修复，
> 因此它是**本地验证证据，不是 release/简历基准**；提交到 clean commit 后必须原样重跑，
> 才能升级为正式数字。

## 1. 环境与边界

| 项 | 值 |
|----|-----|
| 日期 | 2026-08-23 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU（`nvidia-smi` 6144 MiB；CUDA JSON 向下取整为 6143 MiB） |
| Windows 驱动 | 610.88 |
| CUDA Toolkit / runtime | nvcc 12.0.140 / runtime 12000；driver API 13030 |
| 基线 commit | `daeb14f62521cf43a650a70bfb38ea6663dc9d92` |
| 工作区 | dirty：包含本轮正确性、FFI 文档、格式与 benchmark schema v2 修改 |
| 模型 | Qwen2.5-0.5B-Instruct GGUF Q4_K_M；加载后重量化为 W8A16 |
| prompt / 输出 | `你好`（1 token）/ 最多 64 新 token，greedy |
| 每个进程 | 1 次 Graph capture 预热 + 3 warmup + 10 timed iterations |
| 重复 | 3 个独立进程/组，按 on→off、off→on、on→off 交错 |
| 频率 | 未锁频；记录真实 Laptop/WSL2 波动，不宣称数据中心 GPU 结果 |

WSL 默认 loader 被 Linux `libcuda` 包遮蔽，本机命令显式使用
`LD_LIBRARY_PATH=/usr/lib/wsl/lib`。这只修正驱动库选择，不改变 benchmark 参数。

## 2. 复现命令

```bash
MODEL=/home/shane/github/open-infra-ai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf

LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  ./build/tiny_llm_bench "$MODEL" --prompt "你好" \
  --max-tokens 64 --warmup 3 --iters 10 --json

LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  ./build/tiny_llm_bench "$MODEL" --prompt "你好" \
  --max-tokens 64 --warmup 3 --iters 10 --no-graphs --json
```

JSON 中 Graph on 三组均为 `enabled=true,captured=true`；off 三组均为
`enabled=false,captured=false`。每组均生成 64 token。

## 3. 六组原始汇总

| 组 | 顺序 | Graph | TTFT mean / p50 (ms) | TPOT mean / p50 (ms) | decode tok/s | 常驻显存差值 (MB) |
|----|------|-------|----------------------|----------------------|--------------|---------------------|
| 1 | 先 | on | 10.633 / 9.356 | 5.298 / 5.309 | 188.759 | 3368 |
| 1 | 后 | off | 9.906 / 8.862 | 8.866 / 8.765 | 112.791 | 3364 |
| 2 | 先 | off | 8.998 / 9.033 | 8.810 / 8.804 | 113.505 | 3364 |
| 2 | 后 | on | 9.068 / 8.911 | 5.379 / 5.387 | 185.893 | 3368 |
| 3 | 先 | on | 10.536 / 9.047 | 5.323 / 5.318 | 187.877 | 3368 |
| 3 | 后 | off | 10.349 / 8.676 | 8.715 / 8.473 | 114.739 | 3364 |

TTFT mean 在 WSL2 中出现 15–19ms 的尾部抖动，因此聚合 TTFT 使用“每进程 p50 的中位数”；
TPOT 与吞吐使用“每进程 mean 的中位数”。

## 4. 聚合结论

| 指标 | Graph off | Graph on | 变化 |
|------|-----------|----------|------|
| TTFT p50 的跨进程中位数 | 8.862 ms | 9.047 ms | +2.1% |
| TPOT mean 的跨进程中位数 | 8.810 ms | 5.323 ms | **-39.6%** |
| decode tok/s 的跨进程中位数 | 113.505 | 187.877 | **+65.5%** |
| 常驻显存差值 | 3364 MB | 3368 MB | +4 MB（非峰值） |

结论只支持：在这台 RTX 3060 Laptop、这个模型与 64-token decode 上，Graph 重放显著减少
当前实现的 launch/host 调度开销，且 TTFT p50 代价较小。它不支持外推到其他 GPU、模型、
batch 或长上下文，也不能把 +65.5% 写成“整体推理加速 65.5%”。

## 5. 正确性与 profiler 状态

- `InferenceEngineTest.CudaGraphsGenerateMatchesNonGraph` 用真实模型跑 1 项通过，
  on/off 16 token 逐 token 相等，并断言同请求 TTFT 已记录。
- `ncu` 2022.4.1 返回 `ERR_NVGPUCTRPERM`，没有采到性能计数器。
- `nsys profile` 2022.4.2 可生成 `.qdstrm`，但本机缺 importer，不能转换报告或运行 stats。

因此本报告证明端到端 A/B 与输出一致性，尚不能用正式 timeline/warp stall 数据完成因果归因。
后续在云 GPU 或 profiler 权限完整的 Linux 主机上补 nsys/ncu，是把“方向性收益”升级为
“可解释性能证据”的必要步骤。
