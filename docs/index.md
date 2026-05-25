---
layout: home

hero:
  name: "Tiny-LLM"
  text: "CUDA C++ Inference, Kept Small"
  tagline: Focused Transformer inference engine with W8A16 kernels, explicit KV cache management, and a deliberately small repository surface.
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
  - icon: "⚡"
    title: W8A16 Runtime
    details: INT8 weights with FP16 activations for a compact CUDA inference path.
  - icon: "📦"
    title: Narrow Loading Surface
    details: Runtime loading stays on the supported binary path, while GGUF parsing remains available for inspection and validation.
  - icon: "🧠"
    title: Explicit KV Cache
    details: Pre-allocated sequence slots keep autoregressive decoding predictable.
  - icon: "🛡️"
    title: "Result<T> APIs"
    details: Host-side fallible operations return explicit results instead of hiding failures.
  - icon: "🧪"
    title: Tested Core
    details: GoogleTest and RapidCheck cover the core loader, cache, and generation paths.
  - icon: "📚"
    title: Focused Docs
    details: Pages documents the engine itself instead of duplicating process and changelog scaffolding.
---
