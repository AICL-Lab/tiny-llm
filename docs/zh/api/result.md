# Result\<T\>

`Result<T>` 是 Tiny-LLM 在宿主侧使用的可失败返回类型。

## 结构

```cpp
template <typename T>
class Result {
public:
    static Result<T> ok(T value);
    static Result<T> err(std::string message);

    bool isOk() const;
    bool isErr() const;

    T& value();
    const T& value() const;
    const std::string& error() const;
    T valueOr(T default_value) const;
};

template <>
class Result<void> {
public:
    static Result<void> ok();
    static Result<void> err(std::string message);

    bool isOk() const;
    bool isErr() const;
    const std::string& error() const;
};
```

## 常见用法

```cpp
auto engine_result = tiny_llm::InferenceEngine::load("model.bin", config);
if (engine_result.isErr()) {
    std::cerr << engine_result.error() << '\n';
    return 1;
}

auto engine = std::move(engine_result.value());
```

## 使用建议

- 在调用 `value()` 或 `error()` 前，先检查 `isOk()` / `isErr()`。
- 只有在 fallback 真正合理时才使用 `valueOr()`。
- 优先向上传播错误，而不是静默吞掉失败。
