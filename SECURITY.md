# Security Policy

## Reporting a Vulnerability

We take the security of Tiny-LLM seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via GitHub Security Advisories:

1. Go to [Tiny-LLM Security Advisories](https://github.com/LessUp/tiny-llm/security/advisories)
2. Click "Report a vulnerability"
3. Fill out the form with details about the vulnerability

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your message.

### What to Include

Please include the following information in your report:

- **Description**: A clear description of the vulnerability
- **Impact**: What an attacker could achieve by exploiting this vulnerability
- **Reproduction**: Step-by-step instructions to reproduce the issue
- **Environment**: OS, CUDA version, compiler version
- **Proof of Concept**: If available, a minimal code example demonstrating the issue

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Initial Response | Within 48 hours |
| Vulnerability Confirmation | Within 7 days |
| Fix Development | Depends on severity |
| Fix Release | Typically within 14 days of confirmation |

### Disclosure Policy

- We will acknowledge your email within 48 hours
- We will confirm the vulnerability and determine its severity
- We will develop a fix and test it
- We will release the fix and publish a security advisory
- We will credit you in the advisory (unless you prefer to remain anonymous)

## Security Considerations

### Memory Safety

Tiny-LLM uses CUDA and C++ for high-performance inference. While we follow best practices for memory safety:

- RAII pattern for resource management
- Bounds checking in critical paths
- CUDA error checking with `CUDA_CHECK` macro

Users should be aware that:
- Invalid model files could cause buffer overflows
- Maliciously crafted weights could cause undefined behavior
- GPU memory exhaustion is possible with large models

### Best Practices for Users

1. **Model Files**: Only load model files from trusted sources
2. **Input Validation**: Validate prompt inputs before processing
3. **Memory Limits**: Set appropriate memory limits for your GPU
4. **Updates**: Keep Tiny-LLM updated to the latest version

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 2.0.x   | ✅ Active |
| 1.x.x   | ❌ End of Life |

## Security Updates

Security updates will be announced via:
- [GitHub Security Advisories](https://github.com/LessUp/tiny-llm/security/advisories)
- [GitHub Releases](https://github.com/LessUp/tiny-llm/releases)
- [Changelog](CHANGELOG.md)

---

Thank you for helping keep Tiny-LLM and its users safe!
