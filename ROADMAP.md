# Tiny-LLM 路线图

> 目标：把 Tiny-LLM 做成**可在面试中完整讲述的端到端推理引擎作品**——
> 真实模型、真实硬件、可复现数字、有对比基线。
> 原则：每完成一项，更新 README「项目状态」表，把 ❌ 改成 ✅。

## 阶段 0：诚实基线（已完成）

- [x] 清理文档中无法复现的性能数字与不存在的配置项
- [x] README 明确标注已实现/未实现边界

## 阶段 1：端到端跑通真实模型（最高优先级）

- [x] 本机 CUDA 构建环境（CUDA 12.9 toolkit + gcc 13，`cmake --build` 全绿）
- [x] 用真实 GGUF 模型验证加载路径（Qwen2.5-0.5B-Instruct Q4_K_M，469MB，291 tensors）：
      架构感知配置提取、vocab 派生、Q5_0/Q4_K/Q6_K 反量化，与 Python gguf 参考实现一致
- [x] `--inspect` CPU-only 检查模式（无需 GPU 即可验证任意 GGUF 文件）
- [x] 修复加载路径暴露的真实 bug（架构前缀硬编码、vocab_size 来源错误、
      未知量化类型静默假设 FP16），均有回归测试
- [x] 实现 tokenizer（Qwen2.5 为 gpt2 风格 BPE，GGUF 内嵌 tokens+merges，从 GGUF 读取）
      ——手写 Qwen2 预分词正则 + GPT-2 字节编码 + BPE，与 HuggingFace tokenizers
      库 30 例 417 token 逐 id 差分对齐，decode 无损往返
- [ ] GPU 环境验证权重上传与端到端生成（需有卡机器：`tiny_llm_demo model.gguf --prompt "..."`）
- [ ] 与 llama.cpp 同模型同输入对比输出一致性（token 级 diff）

**完成证据**：一条命令从 GGUF 到文本输出，且与 llama.cpp 输出对齐。

## 阶段 2：可复现的性能基准

- [ ] 编写 benchmark 驱动（固定预热/迭代/统计口径，CUDA Event + 宿主计时）
- [ ] 指标：TTFT、TPOT、decode 吞吐（tok/s）、峰值显存
- [ ] 对比基线：llama.cpp（同硬件、同量化、同模型）
- [ ] nsys 时间线 + ncu kernel 分析各一份，归档到 docs/performance/
- [ ] 将实测数字回填 docs/performance/（附硬件/软件版本/复现命令）

**完成证据**：README 出现第一张带基线对比、可复现的性能表。

## 阶段 3：选一个推理加速主题做深（二选一）

**选项 A：CUDA Graphs**
- [ ] decode 阶段 graph capture/replay，配置开关
- [ ] before/after 的 launch 开销与 TPOT 对比数字

**选项 B：Speculative Decoding（草稿模型 n-gram 或小型同构模型）**
- [ ] 接受率统计与端到端加速比
- [ ] 与标准 decode 的正确性差分测试

**完成证据**：一个有数字、有 profiling 证据、能讲清取舍的优化故事。

## 阶段 4：工程完整度（面试加分项）

- [ ] GQA/MQA 模型真实验证（num_kv_heads < num_heads）
- [ ] 长上下文 KV Cache 显存实测曲线
- [ ] 失败路径审计：加载损坏文件、显存不足、超长输入的行为与测试
- [ ] （可选）与 [paged-infer](https://github.com/AICL-Lab/paged-infer) 的调度层对接原型

## 面试讲述要点（完成后自查）

1. 能在 10 分钟内讲清：架构、量化方案、KV Cache 设计、瓶颈与优化过程
2. 每个性能数字都能回答：基线是什么、硬件是什么、怎么复现
3. 能回答"为什么不用 llama.cpp/vLLM"：学习目的、可控性、以及从它们学到的设计
