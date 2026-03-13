---
layout: default
title: "2026-03-13 — 文档与 Pages 规范化"
description: "Tiny-LLM 文档入口、README 职责分离与 Pages workflow 分支触发修正"
---

# 2026-03-13 — 文档与 Pages 规范化

## 变更背景

- 继续推进仓库群 GitHub Pages 与文档信息架构标准化。
- 此前 `README.md`、`README.zh-CN.md` 与 `index.md` 都承担了较完整的项目说明，仓库入口与文档入口重复明显。
- 同时，Pages workflow 只监听 `main`，而仓库当前分支为 `master`，会导致文档修改无法自动触发部署。

## 导航与目录调整

- 保持现有 Jekyll 目录结构不变，继续使用根目录 `index.md` 作为站点首页，`docs/API.md` 作为 API 参考页，`changelog/` 作为归档入口。
- 文档入口层级调整为：首页负责导读，API / 贡献指南 / 更新日志分别承担参考、开发与归档职责。

## 首页调整

- `README.md` / `README.zh-CN.md` 收敛为仓库入口，只保留项目定位、最小构建命令和文档链接。
- `index.md` 改写为文档首页，新增项目定位、适合谁、从哪里开始、推荐阅读路径与核心文档表。
- 首页不再重复完整架构细节和长篇项目结构说明，把这些内容下沉到 API、贡献指南和 changelog 中。

## Pages / Workflow 调整

- `.github/workflows/pages.yml` 的推送触发分支从仅 `main` 扩展为 `master, main`。
- 保持现有 sparse checkout、Jekyll build 和 artifact 上传逻辑不变，避免在修正文档入口时引入新的部署风险。

## 验证结果

- 人工检查 `README` 与 `index.md` 职责已分离：README 只承担仓库入口，站点首页只承担文档入口。
- 人工检查首页到 `docs/API`、`CONTRIBUTING`、`changelog/` 的链接均与现有文件结构一致。
- 已确认当前仓库分支为 `master`，Pages workflow 现已覆盖实际使用分支。
- 本次未运行本地 Jekyll 构建；后续可在具备 Ruby / Jekyll 环境时补充静态构建验证。

## 后续待办

- 视后续文档扩展情况，为 `docs/` 增加更多参考页时继续保持首页只做导读。
- 如后续新增更多 changelog 条目，继续维护 `changelog/index.md` 的时间倒序索引。
