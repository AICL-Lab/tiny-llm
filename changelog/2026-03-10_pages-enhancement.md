---
title: "2026-03-10 — GitHub Pages 完善"
---

# 2026-03-10 — GitHub Pages 完善

## 文档页面 Jekyll 集成

1. **`docs/API.md` 添加 YAML frontmatter** — title + description，Jekyll 正确渲染为独立页面
2. **`CONTRIBUTING.md` 添加 YAML frontmatter** — 同上
3. **`docs/API.md` 添加导航页脚** — 返回首页 / 更新日志 / 贡献指南
4. **`CONTRIBUTING.md` 添加导航页脚** — 返回首页 / API 参考 / 更新日志

## Jekyll 配置补全 (`_config.yml`)

5. **补全 exclude 列表** — 新增 `.vscode`、`.cache`、`.gitattributes`，防止非文档文件被 Jekyll 处理

## 首页增强 (`index.md`)

6. **添加 CI/Pages 徽章** — 与 README 保持一致
7. **切换为中文文案** — 核心特性、采样策略、快速开始等 section 使用中文
8. **添加"工程质量"特性项** — CI 流水线、clang-format、RAII、Result 错误处理
9. **扩展架构图** — 新增 StreamPool、KV Cache Manager、Result 错误处理、Elementwise/Warp Utilities 层
10. **添加完整项目结构** — 列出所有头文件、kernel、源文件及其功能描述
11. **添加"性能优化"表格** — 6 项优化技术：tiling、warp shuffle、融合反量化、合并访问、预分配、多流
12. **添加"GPU 架构支持"表格** — Volta → Hopper 全系列
13. **扩展测试 section** — 增加 Transformer、Integration 测试套件
14. **添加"技术栈"表格** — 语言、构建、GPU、量化、测试、CI、代码风格
15. **使用示例改进** — 添加 Result 错误处理（isErr 检查）

## README 增强

16. **`README.md` 添加 CMake 徽章** — 补齐与中文版一致
17. **`README.md` 添加架构图** — ASCII 架构图 + 扩展组件描述
18. **`README.md` 添加 GPU 支持表** — Volta → Hopper
19. **`README.md` 添加测试详情表** — 6 个测试套件覆盖内容
20. **`README.md` 扩展项目结构** — 完整文件列表含描述
21. **`README.zh-CN.md` 添加架构图** — 与 README.md / index.md 保持同步
22. **`README.zh-CN.md` 添加性能优化表** — 6 项优化技术
23. **`README.zh-CN.md` 添加 GPU 架构支持表** — Volta → Hopper
24. **`README.zh-CN.md` 添加技术栈表** — CI、代码风格等
25. **`README.zh-CN.md` 扩展项目结构** — 完整文件列表含描述

## 工作流与杂项

26. **`pages.yml` sparse-checkout 修正** — `docs` → `docs/`、`changelog` → `changelog/`（规范目录路径）
27. **`.gitignore` 添加 `.cache/`** — 排除 clangd 缓存目录
