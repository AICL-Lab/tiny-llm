# CUDA Graphs：decode 阶段捕获/重放

> 目标：消除 decode 阶段每次 token 的 kernel launch 开销（24 层 × 每层
> 约 10+ kernel ≈ 250+ 次 launch/step）。本文记录设计、capture 范围、
> before/after 数字、失败原因与 fallback 行为。

## 1. 开关

- 环境变量 `TLLM_CUDA_GRAPHS`：**默认开启**（任务 C2 起）；`TLLM_CUDA_GRAPHS=0`
  显式关闭（opt-out）。
- `tiny_llm_bench` 默认跑 graphs 路径；`--no-graphs` 等价于设置
  `TLLM_CUDA_GRAPHS=0`（A/B 对比用），`--graphs` 为兼容/诊断保留（打印当前
  状态，默认已开启）。
- `InferenceEngine` 构造时读取；捕获失败自动回退并 `TLLM_WARN` 记录原因。

```bash
./build/tiny_llm_demo model.gguf --prompt "你好" --max-tokens 32 --show-tokens   # 默认 graphs 开
TLLM_CUDA_GRAPHS=0 ./build/tiny_llm_demo model.gguf --prompt "你好" --max-tokens 32 --show-tokens
diff <(./build/tiny_llm_demo ...) <(TLLM_CUDA_GRAPHS=0 ./build/tiny_llm_demo ...)  # 必须逐 token 一致
```

## 2. 设计：把 step 变化值移出 kernel 参数

CUDA Graph 捕获会把 kernel 参数（含指针与标量）固化为图节点。decode 路径
中每个 step 都在变化的值若作为参数，重放时就会过期。因此先把它们改为
**从 device 端 int 读取**（host 每次 decode 前在同一 stream 上
`cudaMemcpyAsync` 更新）：

| 变化值 | kernel/调用 | 间接化 |
|--------|------------|--------|
| 可见 KV 长度 `visible_len` | `attention_decode` | device int（任务 3.1） |
| append 写位置 `write_pos` | `appendKV` → `append_kv_at` kernel | device int（任务 3.2） |
| RoPE 起始位置 `start_position` | `apply_rope_inplace` | device int（任务 3.2） |
| token id 输入 | `embedTokens` | 固定缓冲 `graph_token_` |
| 激活行地址 | decode step | 固定用 `hidden_states_` 最后一行 |

> 关键点：append 的目标地址不能由 host 计算后传入 —— 那会被 graph 固化，
> 重放时写回同一 slot 而不是推进。因此 append 改为一个读 device 写位置的
> 小 kernel（`append_kv_at`，见 `kernels/elementwise.cu`）。

## 3. capture 范围

只捕获确定性的 device 工作（同一函数 `runDecodeDevicePath`，graph 与直接
执行共用，保证两条路径一致）：

```
embedTokens(graph_token_, 1, 固定行)
  → 24 层 TransformerLayer::forward
      （rmsnorm / qkv 投影 / RoPE(device pos) / append_kv_at(device pos) /
        attention_decode(device visible_len) / out 投影 / FFN）
  → finalNorm → computeLogits（写 logits_）
```

**不捕获**（host 侧操作）：
- `advanceSeqLen`（host 计数器推进）
- logits D2H copy、采样（greedy/temperature/top-k/top-p）
- `sampleFromHidden` 的首 token 路径（来自 prefill，非 decode）

## 4. 捕获时机与重放

1. **第一次 decode step**：直接执行 device 序列并 `cudaStreamSynchronize`，
   得到本步结果；随后在 `cudaStreamCaptureModeThreadLocal` 捕获区内**记录**
   一次（capture 不真正执行 kernel）。`cudaStreamEndCapture` +
   `cudaGraphInstantiate`。**本步结果来自直接执行，不重复 launch。**
2. **之后每个 decode step**：
   - host 更新 device 值：`decode_len_`（可见长度）、`rope_pos_`（位置）、
     `append_pos_`（写位置）、`graph_token_`（token id）
   - `cudaGraphLaunch`
   - host 侧 `advanceSeqLen` → D2H 读 logits → 采样

同一引擎多次 `generate()` 复用已捕获 graph：KV cache 池地址稳定
（max_batch_size=1，slot 0 固定），每次重新分配序列后 device 值会重置，
重放仍正确。

## 5. 数值一致性

`attention`、`W8A16` 等所有 kernel 在重放前后数值逐 token 一致（greedy）。
差分测试：

- `tests/test_inference_engine.cu::CudaGraphsGenerateMatchesNonGraph`
  （门控 `TLLM_GGUF_TEST_MODEL`）：同一模型 graphs 开/关 generate 16 token，
  逐 token 断言相等。
- 手工验收：默认（graphs 开）与 `TLLM_CUDA_GRAPHS=0` 输出 diff（见第 1 节）。

如果出现不一致，先 `TLLM_CUDA_GRAPHS=0` 排除 graph 引入，再检查是否有未
固定地址的缓冲（如每个 step 重新 cudaMalloc 的输入缓冲）或 host 端顺序
依赖。

## 6. 失败原因与 fallback

capture 或实例化阶段任何失败都会 **关闭 graphs 并回退常规路径**（不改变
正确性）：

- `cudaStreamBeginCapture` 返回错误（stream 非空闲 / 不支持）
- capture 区内任何 CUDA 调用失败（含 `runDecodeDevicePath` 内 kernel 启动
  失败抛出的 `CudaException`，用 try/catch 捕获）
- `cudaStreamEndCapture` 失败
- `cudaGraphInstantiate` 失败（参数不合法 / 资源不足）

回退时 `TLLM_WARN` 记录原因；首次 decode 的直接执行结果仍被使用，后续
步骤走直接路径，输出与 graphs=0 完全一致。

已知限制：本实现捕获的是**单 token decode**；prefill、FFI（`ffi.cpp`）路径
不使用 graphs。若在 capture 中调用 `cudaMalloc`（如每次 decode 临时分配
输入缓冲）会导致失败 —— 因此 decode 输入缓冲是构造期固定的成员
（`graph_token_`）。

## 7. before/after 数字

### 7.1 实现当时的历史快照（C1 转置快路径之前）

环境：RTX 3060 Laptop（6GB），CUDA 12.0，Qwen2.5-0.5B-Instruct GGUF Q4_K_M
（本仓库重量化 W8A16），prompt "你好"，greedy，max_tokens=64，
warmup 5 / iters 20，`tiny_llm_bench` 墙钟。

| 指标 | graphs=0 | graphs=1 | 变化 |
|------|----------|----------|------|
| TPOT (ms/token) | 24.28 | 22.62 | **-6.8%** |
| decode tok/s | 41.18 | 44.20 | **+7.3%** |
| TTFT (ms) | 24.94 | 28.03 | +3.1ms（首次 decode 含捕获开销） |

> 说明：这是 commit `a2a9c58`、C1 转置快路径之前的历史 schema v1 快照，不能与当前
> 结果直接合并。TTFT 略增是首次 decode 的捕获
> 一次性开销（直接执行 + capture 记录 + 实例化），摊薄到长生成可忽略。
> 数字受系统负载影响，建议同环境复测。

commit：`a2a9c58`（任务 3.2 实现）。复现命令见 `benchmark-methodology.md`。

### 7.2 2026-08-23 current-worktree schema v2 验证

schema v2 改为在**同一次请求**内记录 TTFT/TPOT，并明确输出 Graph 实际状态。
RTX 3060 Laptop 上 3 个独立进程、每进程 10 次 timed iteration 的聚合结果：

| 指标 | Graph off | Graph on | 变化 |
|------|-----------|----------|------|
| TTFT p50 的跨进程中位数 | 8.862 ms | 9.047 ms | +2.1% |
| TPOT mean 的跨进程中位数 | 8.810 ms | 5.323 ms | **-39.6%** |
| decode tok/s 的跨进程中位数 | 113.505 | 187.877 | **+65.5%** |

工作区是 dirty，故本表只作为本地验证，不替换 clean commit 的简历基准。完整环境、六组原始
汇总、常驻显存口径与 profiler 限制见
[`results/2026-08-23-cuda-graphs-ab.md`](results/2026-08-23-cuda-graphs-ab.md)。
