# Result\<T\>

`Result<T>` is Tiny-LLM's host-side fallible return type.

## Shape

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

## Typical Usage

```cpp
auto engine_result = tiny_llm::InferenceEngine::load("model.bin", config);
if (engine_result.isErr()) {
    std::cerr << engine_result.error() << '\n';
    return 1;
}

auto engine = std::move(engine_result.value());
```

## Guidance

- Check `isOk()` / `isErr()` before calling `value()` or `error()`.
- Use `valueOr()` only when a fallback is genuinely correct.
- Propagate errors upward instead of swallowing them.
