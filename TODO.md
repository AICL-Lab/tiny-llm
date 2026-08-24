# TODO 待办（2026-08-21 Bug 审计修复收尾）

> 来源：2026-08-21 全仓 Bug 审计（20 项）修复会话。大部分已修复并合入
> （见 `CHANGELOG.md` 与 git log）。T1 已完成；其余内容是保留的审计结论，
> 不是当前优先级。
>
> 当前工作状态以 `README.md`、`ROADMAP.md` 和性能结果目录为准；
> `DEVELOPMENT_PLAN.md` 已标为历史实施快照，不应用其旧测试数量或“未实现”描述。

## 历史工作区状态（2026-08-22）

2026-08-22 GPU 恢复后完成全量验证并分主题提交（T1 完成）：
`ctest --test-dir build --output-on-failure --timeout 300` 非门控 **184/184 通过**
（含 3 个新 CUDA 测试从"从未运行"转为 PASS），9 个真实模型门控测试仍按门控跳过；
6 个分主题 commit 已推送，见 git log 与 CHANGELOG.md。

---

## 已完成任务 T1：驱动恢复后全量回归 + 分主题提交

**背景**：上述未提交改动的 3 个新 CUDA 测试从未实际运行过（TDD 的 RED/GREEN
运行时验证被环境阻断），必须先回归再提交。

**验收步骤**：

```bash
# 0) 先确认 GPU 恢复（Windows 侧 wsl --shutdown 后重进 WSL）
nvidia-smi   # 应正常输出

# 1) 全量构建 + 回归
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure --timeout 300
# 预期：178 + 3 新测试全部通过（9 个门控跳过不变）

# 2) 重点确认 3 个新测试由"从未运行"转为 PASS：
./build/tiny_llm_tests --gtest_filter='*KeyProjectionUsesItsOwnGroupSize*'
./build/tiny_llm_tests --gtest_filter='*SoftmaxHandlesSeqLenBeyondSharedMemoryLimit*'
./build/tiny_llm_tests --gtest_filter='*InvalidBlockIdIsGuardedNotDereferenced*'
```

**提交拆分建议**（单主题单提交，沿用 conventional commits 风格）：

1. `fix(transformer): use per-weight group_size in all projections`（src/transformer.cpp + tests/test_transformer.cu）
2. `fix(kernel): softmax with O(1) shared memory`（kernels/attention.cu + tests/test_kernels.cu 中 softmax 测试）
3. `fix(kernel): guard paged block_id range`（kernels/paged_kv.* + src/transformer.cpp 调用点 + 对应测试）
4. `fix(parser): sanity-limit GGUF entry counts`（src/gguf_parser.cpp + tests/test_model_loader.cpp）
5. `fix(tokenizer): per-byte fallback on invalid UTF-8 continuation`（src/tokenizer.cpp、include/tiny_llm/tokenizer.h + tests/test_tokenizer.cpp）
6. `chore(build/misc): cuda arch 70, sizeof semantics, drop redundant cast`（CMakeLists.txt、src/model_loader.cpp、src/quantization.cpp）

---

## 任务 T2（可选）：审计遗留观察项

以下两项经复核**明确不修**，留档防止重复排查：

- **tiled GEMM 1024 线程/块**（`w8a16_matmul.cu:395`）：在 CUDA 合法上限内、
  共享内存 ~12KB、历史全绿运行已验证 launch 成功。改动有性能回归风险，按设计保留。
- **全零组 scale=1.0**（`quantization.cpp:248`）：反量化 0×1=0 数值正确；
  改 scale=0 反而在量化循环引入除零风险。

若未来做 kernel 优化专项，可重新评估 tiled GEMM 的寄存器压力（ncu 数据可用后）。


---

## 任务 T3（可选）：2026-08-22 全仓复核后明确不修的项

以下三项经再次核实**明确不修**，留档防止重复排查：

- **repetition penalty 在 fp16 上施加**（`inference_engine.cpp` 采样路径）：
  logits 本就是 fp16，惩罚值经 float 计算后写回 half 的往返误差 <0.1%，对采样
  分布影响可忽略；修复需将 `penalized` 缓冲及 `sampleGreedy/Temperature/TopK/TopP`
  全族改为 float，重构面大、收益微、回归风险高。触发条件：logits 改 fp32 输出或
  追求与 llama.cpp 逐位对齐时再评估。
- **`tinyllm_load` 对 GGUF 二次解析**（`src/ffi.cpp`）：`parser.parse()` 已提取
  config，`loadGGUF` 内部又完整 parse 一次，属纯启动性能浪费；修复需给
  `ModelLoader::loadGGUF` 增加"接受已 parse 的 parser"重载，接口侵入大。触发条件：
  大模型（>10GB GGUF）启动延迟成为瓶颈时再优化。
- **`samplingRng(seed≠0)` 逐 token 重播种**（`inference_engine.cpp:samplingRng`）：
  `seed != 0` 时每 token 用同一随机序列（每步采样退化）；但当前所有调用方均传默认
  `seed=0`（一次性初始化、跨步演进），实际行为无害。触发条件：新增显式固定 seed
  采样的调用方（如可复现采样）前，应改为"仅在首次/显式 reset 时播种"。
