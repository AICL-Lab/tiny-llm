# TODO 待办（2026-08-21 Bug 审计修复收尾）

> 来源：2026-08-21 全仓 Bug 审计（20 项）修复会话。大部分已修复并合入
> （见 `CHANGELOG.md` 与 git log），本页只记录**剩余待办**。
> 执行协议沿用 `DEVELOPMENT_PLAN.md` 第 0 节：单任务单提交、验收命令全绿。

## ⚠️ 当前工作区状态（下次开发先读这个）

2026-08-22 GPU 恢复后完成全量验证并分主题提交（T1 完成）：
`ctest --test-dir build --output-on-failure --timeout 300` 非门控 **184/184 通过**
（含 3 个新 CUDA 测试从"从未运行"转为 PASS），9 个真实模型门控测试仍按门控跳过；
6 个分主题 commit 已推送，见 git log 与 CHANGELOG.md。

---

## 任务 T1：驱动恢复后全量回归 + 分主题提交（最高优先级）

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
