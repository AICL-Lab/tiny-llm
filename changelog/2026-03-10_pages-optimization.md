---
title: "2026-03-10 — GitHub Pages 优化"
---

# 2026-03-10 — GitHub Pages 优化

## Jekyll 配置增强 (`_config.yml`)

1. **添加 `repository` 和 `show_downloads`** — 完善 Cayman 主题配置
2. **添加 SEO 元数据** — `url`/`baseurl`/`lang`/`author`，改善搜索引擎索引
3. **添加 `jekyll-seo-tag` 插件** — 自动注入 Open Graph / Twitter Card 元数据
4. **添加 kramdown GFM 设置** — 明确 `markdown: kramdown` + `input: GFM` + `syntax_highlighter: rouge`
5. **添加 `defaults`** — 全局默认 `layout: default`，changelog 目录也默认使用 default 布局
6. **添加 `exclude` 列表** — 排除源代码（`include`/`kernels`/`src`/`tests`）、构建配置、工具配置等

## 工作流优化 (`pages.yml`)

7. **添加 sparse checkout** — 仅检出文档相关文件，跳过 C++/CUDA 源代码
8. **添加 `sparse-checkout-cone-mode: false`** — 修复 cone 模式下文件级匹配问题
9. **细化 `paths` 触发条件** — 从通配 `'*.md'` 改为显式列出文档文件路径，添加 `changelog/**`
10. **添加 job names** — `Build Pages` / `Deploy Pages`

## Changelog 可浏览化

11. **新建 `changelog/index.md`** — 更新日志索引页，按时间倒序排列
12. **3 个 changelog `.md` 添加 YAML frontmatter** — 使 Jekyll 将其渲染为正式页面

## 主页改进 (`index.md`)

13. **添加 SEO 描述** — frontmatter `title` + `description` 字段
14. **添加"最近更新"section** — 展示最近 3 条变更摘要 + "查看完整更新日志"链接
15. **重构文档链接** — 从英文 Documentation 改为中文"文档"section，增加更新日志和贡献指南入口

## README 增强

16. **`README.md` + `README.zh-CN.md` 添加 CI/Pages 徽章** — 链接到 GitHub Actions 和项目主页
17. **添加 Project Page / 项目主页链接** — 语言切换行增加项目主页入口
