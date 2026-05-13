---
layout: home

hero:
  name: "Tiny-LLM"
  text: "教育性 CUDA LLM 推理引擎"
  tagline: 一个极简的教育性 LLM 推理引擎，使用 CUDA C++ 实现，支持 INT8 量化
  image:
    src: /images/logo.svg
    alt: Tiny-LLM Logo
  actions:
    - theme: brand
      text: 开始使用
      link: /zh/guide/getting-started
    - theme: alt
      text: GitHub 仓库
      link: https://github.com/shane0/tiny-llm

features:
  - icon: "\U0001F680"
    title: 高性能
    details: 优化的 CUDA 内核，用于注意力机制、前馈网络和 W8A16 量化推理
  - icon: "\U0001F4DA"
    title: 教育性
    details: 清晰、文档完善的代码库，专为学习 LLM 推理内部原理而设计
  - icon: "\U0001F4CA"
    title: INT8 量化
    details: 每组 INT8 量化，在最小化精度损失的同时提高内存效率
  - icon: "\U0001F512"
    title: 无异常设计
    details: 基于 Result\<T\> 的错误传播，实现可预测且安全的错误处理
  - icon: "\U0001F9E0"
    title: KV 缓存
    details: 预分配的 KV 缓存管理器，用于高效的自回归生成
  - icon: "\U0001F310"
    title: 跨平台
    details: 跨平台支持，使用 CMake 构建系统，支持 Linux 和 Windows
---
