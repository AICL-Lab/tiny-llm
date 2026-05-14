import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

// https://vitepress.dev/reference/site-config
export default withMermaid(
  defineConfig({
    title: 'Tiny-LLM',
    description: 'CUDA-native inference engine for Transformer models with W8A16 quantization',

    // Base URL for GitHub Pages deployment
    base: '/tiny-llm/',

    // Clean URLs without .html extension
    cleanUrls: true,

    // Last updated timestamp
    lastUpdated: true,

    // Head configuration
    head: [
      ['meta', { name: 'theme-color', content: '#00D4AA' }],
      ['meta', { name: 'og:type', content: 'website' }],
      ['meta', { name: 'og:title', content: 'Tiny-LLM | Technical Whitepaper' }],
      ['meta', { name: 'og:description', content: 'CUDA-native inference engine for Transformer models with W8A16 quantization' }],
      ['link', { rel: 'icon', href: '/tiny-llm/favicon.svg' }],
      ['link', { rel: 'apple-touch-icon', href: '/tiny-llm/apple-touch-icon.png' }],
      // Google Fonts
      ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
      ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
      ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap' }],
    ],

    // Markdown configuration
    markdown: {
      theme: {
        light: 'github-light',
        dark: 'github-dark',
      },
      lineNumbers: true,
    },

    // Locales for i18n
    locales: {
      root: {
        label: 'English',
        lang: 'en',
        themeConfig: {
          nav: [
            { text: 'Architecture', link: '/en/architecture/' },
            { text: 'Performance', link: '/en/performance/' },
            { text: 'API', link: '/en/api/' },
            { text: 'Guide', link: '/en/guide/getting-started' },
            { text: 'Changelog', link: '/en/changelog/' },
          ],
          sidebar: {
            '/en/architecture/': [
              {
                text: 'Architecture',
                items: [
                  { text: 'Overview', link: '/en/architecture/' },
                  { text: 'Inference Engine', link: '/en/architecture/inference-engine' },
                  { text: 'W8A16 Quantization', link: '/en/architecture/quantization' },
                  { text: 'KV Cache Design', link: '/en/architecture/kv-cache' },
                  { text: 'CUDA Kernels', link: '/en/architecture/cuda-kernels' },
                  { text: 'Memory Model', link: '/en/architecture/memory-model' },
                ],
              },
            ],
            '/en/performance/': [
              {
                text: 'Performance',
                items: [
                  { text: 'Overview', link: '/en/performance/' },
                  { text: 'Benchmarks', link: '/en/performance/benchmarks' },
                  { text: 'Profiling Guide', link: '/en/performance/profiling' },
                  { text: 'Optimization', link: '/en/performance/optimization' },
                ],
              },
            ],
            '/en/api/': [
              {
                text: 'API Reference',
                items: [
                  { text: 'Overview', link: '/en/api/' },
                  { text: 'InferenceEngine', link: '/en/api/inference-engine' },
                  { text: 'ModelConfig', link: '/en/api/model-config' },
                  { text: 'Result<T>', link: '/en/api/result' },
                  { text: 'KVCacheManager', link: '/en/api/kv-cache' },
                  { text: 'References', link: '/en/api/references' },
                ],
              },
            ],
            '/en/guide/': [
              {
                text: 'Guide',
                items: [
                  { text: 'Getting Started', link: '/en/guide/getting-started' },
                  { text: 'Installation', link: '/en/guide/installation' },
                  { text: 'Quick Start', link: '/en/guide/quickstart' },
                  { text: 'Configuration', link: '/en/guide/configuration' },
                  { text: 'Quantization', link: '/en/guide/quantization' },
                  { text: 'Troubleshooting', link: '/en/guide/troubleshooting' },
                ],
              },
            ],
            '/en/contributing/': [
              {
                text: 'Contributing',
                items: [
                  { text: 'Developer Guide', link: '/en/contributing/' },
                ],
              },
            ],
            '/en/changelog/': [
              {
                text: 'Changelog',
                items: [
                  { text: 'Overview', link: '/en/changelog/' },
                  { text: 'v2.0.2', link: '/en/changelog/v2.0.2' },
                  { text: 'v2.0.1', link: '/en/changelog/v2.0.1' },
                  { text: 'v2.0.0', link: '/en/changelog/v2.0.0' },
                ],
              },
            ],
          },
          editLink: {
            pattern: 'https://github.com/AICL-Lab/tiny-llm/edit/master/docs/:path',
            text: 'Edit this page on GitHub',
          },
          footer: {
            message: 'Released under the MIT License.',
            copyright: 'Copyright © 2024-present Tiny-LLM Contributors',
          },
          docFooter: {
            prev: 'Previous',
            next: 'Next',
          },
          outline: {
            label: 'On this page',
          },
          lastUpdated: {
            text: 'Last updated',
            formatOptions: {
              dateStyle: 'medium',
              timeStyle: 'short',
            },
          },
          langMenuLabel: 'Language',
          returnToTopLabel: 'Return to top',
          sidebarMenuLabel: 'Menu',
          darkModeSwitchLabel: 'Theme',
          lightModeSwitchTitle: 'Switch to light theme',
          darkModeSwitchTitle: 'Switch to dark theme',
        },
      },
      zh: {
        label: '简体中文',
        lang: 'zh-CN',
        link: '/zh/',
        themeConfig: {
          nav: [
            { text: '架构', link: '/zh/architecture/' },
            { text: '性能', link: '/zh/performance/' },
            { text: 'API', link: '/zh/api/' },
            { text: '指南', link: '/zh/guide/getting-started' },
            { text: '更新日志', link: '/zh/changelog/' },
          ],
          sidebar: {
            '/zh/architecture/': [
              {
                text: '架构',
                items: [
                  { text: '概述', link: '/zh/architecture/' },
                  { text: '推理引擎', link: '/zh/architecture/inference-engine' },
                  { text: 'W8A16 量化', link: '/zh/architecture/quantization' },
                  { text: 'KV 缓存设计', link: '/zh/architecture/kv-cache' },
                  { text: 'CUDA 内核', link: '/zh/architecture/cuda-kernels' },
                  { text: '内存模型', link: '/zh/architecture/memory-model' },
                ],
              },
            ],
            '/zh/performance/': [
              {
                text: '性能',
                items: [
                  { text: '概述', link: '/zh/performance/' },
                  { text: '基准测试', link: '/zh/performance/benchmarks' },
                  { text: '分析指南', link: '/zh/performance/profiling' },
                  { text: '优化', link: '/zh/performance/optimization' },
                ],
              },
            ],
            '/zh/api/': [
              {
                text: 'API 参考',
                items: [
                  { text: '概述', link: '/zh/api/' },
                  { text: 'InferenceEngine', link: '/zh/api/inference-engine' },
                  { text: 'ModelConfig', link: '/zh/api/model-config' },
                  { text: 'Result<T>', link: '/zh/api/result' },
                  { text: 'KVCacheManager', link: '/zh/api/kv-cache' },
                  { text: '参考资料', link: '/zh/api/references' },
                ],
              },
            ],
            '/zh/guide/': [
              {
                text: '指南',
                items: [
                  { text: '入门指南', link: '/zh/guide/getting-started' },
                  { text: '安装', link: '/zh/guide/installation' },
                  { text: '快速开始', link: '/zh/guide/quickstart' },
                  { text: '配置', link: '/zh/guide/configuration' },
                  { text: '量化', link: '/zh/guide/quantization' },
                  { text: '故障排除', link: '/zh/guide/troubleshooting' },
                ],
              },
            ],
            '/zh/contributing/': [
              {
                text: '贡献',
                items: [
                  { text: '开发者指南', link: '/zh/contributing/' },
                ],
              },
            ],
            '/zh/changelog/': [
              {
                text: '更新日志',
                items: [
                  { text: '概述', link: '/zh/changelog/' },
                  { text: 'v2.0.2', link: '/zh/changelog/v2.0.2' },
                  { text: 'v2.0.1', link: '/zh/changelog/v2.0.1' },
                  { text: 'v2.0.0', link: '/zh/changelog/v2.0.0' },
                ],
              },
            ],
          },
          editLink: {
            pattern: 'https://github.com/AICL-Lab/tiny-llm/edit/master/docs/:path',
            text: '在 GitHub 上编辑此页',
          },
          footer: {
            message: '基于 MIT 许可证发布',
            copyright: '版权所有 © 2024至今 Tiny-LLM 贡献者',
          },
          docFooter: {
            prev: '上一页',
            next: '下一页',
          },
          outline: {
            label: '页面导航',
          },
          lastUpdated: {
            text: '最后更新于',
            formatOptions: {
              dateStyle: 'medium',
              timeStyle: 'short',
            },
          },
          langMenuLabel: '语言',
          returnToTopLabel: '返回顶部',
          sidebarMenuLabel: '菜单',
          darkModeSwitchLabel: '主题',
          lightModeSwitchTitle: '切换到浅色模式',
          darkModeSwitchTitle: '切换到深色模式',
        },
      },
    },

    // Theme configuration (shared across locales)
    themeConfig: {
      logo: '/tiny-llm/logo.svg',
      siteTitle: 'Tiny-LLM',
      socialLinks: [
        { icon: 'github', link: 'https://github.com/AICL-Lab/tiny-llm' },
      ],
      search: {
        provider: 'local',
        options: {
          translations: {
            button: {
              buttonText: 'Search',
              buttonAriaLabel: 'Search',
            },
            modal: {
              noResultsText: 'No results for',
              resetButtonTitle: 'Reset search',
              footer: {
                selectText: 'to select',
                navigateText: 'to navigate',
                closeText: 'to close',
              },
            },
          },
        },
      },
    },

    // Mermaid configuration
    mermaid: {
      theme: 'base',
      themeVariables: {
        primaryColor: '#00D4AA',
        primaryTextColor: '#fff',
        primaryBorderColor: '#00C49A',
        lineColor: '#76B900',
        secondaryColor: '#F59E0B',
        tertiaryColor: '#8B5CF6',
      },
    },

    // Build options
    build: {
      chunkSizeWarningLimit: 1500,
    },

    // Vite config
    vite: {
      build: {
        minify: 'terser',
        chunkSizeWarningLimit: 1500,
      },
    },
  })
)

// Plugin: llmstxt for LLM-friendly documentation
llmstxt({
  domain: 'https://aicl-lab.github.io/tiny-llm',
  title: 'Tiny-LLM Documentation',
  description: 'CUDA-native inference engine for Transformer models with W8A16 quantization',
  sections: {
    'Architecture': {
      title: 'Architecture',
      description: 'System architecture and design',
    },
    'Performance': {
      title: 'Performance',
      description: 'Benchmarks and optimization',
    },
    'API': {
      title: 'API Reference',
      description: 'Complete API documentation',
    },
    'Guide': {
      title: 'Guide',
      description: 'Getting started and usage guide',
    },
  },
})
