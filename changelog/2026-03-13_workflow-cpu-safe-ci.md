# Workflow CPU-safe CI 调整

日期：2026-03-13

## 变更内容

- 将 `.github/workflows/ci.yml` 从 GitHub Hosted Runner 上的 CUDA 容器构建，调整为仅保留 CPU-safe 的 `Format Check`
- 为 CI 补回 `push`、`pull_request` 与 `workflow_dispatch` 触发，避免只有手动触发却长期红灯
- 统一使用 `jidicula/clang-format-action` 执行格式校验，并排除 `build`、`third_party`、`external`、`vendor` 等目录

## 背景

该仓库原先的 CI 依赖 `nvidia/cuda` 容器，但 GitHub Hosted Runner 不提供可用 GPU，导致工作流长期处于无效或失败状态。本次调整将主线检查收敛为稳定可通过的静态格式校验。
