---
layout: home

hero:
  name: "Tiny-LLM"
  text: "Educational CUDA LLM Inference"
  tagline: A minimal, educational LLM inference engine in CUDA C++ with INT8 quantization
  image:
    src: /images/logo.svg
    alt: Tiny-LLM Logo
  actions:
    - theme: brand
      text: Get Started
      link: /en/guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/AICL-Lab/tiny-llm

features:
  - icon: "\U0001F680"
    title: High Performance
    details: Optimized CUDA kernels for attention, FFN, and quantized inference with W8A16 quantization
  - icon: "\U0001F4DA"
    title: Educational
    details: Clean, well-documented codebase designed for learning LLM inference internals
  - icon: "\U0001F4CA"
    title: INT8 Quantization
    details: Per-group INT8 quantization for memory efficiency with minimal accuracy loss
  - icon: "\U0001F512"
    title: No-Exception Design
    details: Result\<T\> based error propagation for predictable and safe error handling
  - icon: "\U0001F9E0"
    title: KV Cache
    details: Pre-allocated KV cache manager for efficient autoregressive generation
  - icon: "\U0001F310"
    details: Cross-platform support with CMake build system for Linux and Windows
---

<style>
/* Custom home page styles */
.VPHero .name {
  background: linear-gradient(135deg, #00D4AA 0%, #76B900 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.VPHero .image-bg {
  background: linear-gradient(135deg, #00D4AA, #76B900);
  opacity: 0.2;
  filter: blur(100px);
}
</style>
