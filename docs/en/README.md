---
layout: default
title: "Documentation"
description: "Tiny-LLM Documentation Home"
nav_order: 0
---

# Tiny-LLM Documentation

Welcome to the Tiny-LLM documentation! Tiny-LLM is a lightweight CUDA/C++ inference engine for LLMs with W8A16 quantization and efficient KV Cache management.

---

## Getting Started

New to Tiny-LLM? Start here:

| Document | Description |
|----------|-------------|
| [Quick Start](QUICKSTART) | Installation and basic usage |
| [Architecture](ARCHITECTURE) | System design and components |
| [API Reference](API) | Complete API documentation |

---

## Features

- **W8A16 Quantization**: ~50% memory reduction with INT8 weights
- **KV Cache**: Efficient incremental decoding
- **High-Performance Kernels**: CUDA optimizations
- **Multiple Sampling**: Greedy, Temperature, Top-k, Top-p
- **Comprehensive Tests**: GoogleTest + RapidCheck

---

## Project Links

- [GitHub Repository](https://github.com/LessUp/tiny-llm)
- [Changelog](../../changelog/)
- [Contributing Guidelines](../../CONTRIBUTING)

---

## Support

- [GitHub Issues](https://github.com/LessUp/tiny-llm/issues) for bug reports
- [GitHub Discussions](https://github.com/LessUp/tiny-llm/discussions) for questions

---

**Languages**: [English](./) | [中文](../zh/)

[← Home](../../)
