# Result\<T\>

The `Result\<T\>` class provides no-exception error handling for Tiny-LLM operations.

## Overview

Instead of throwing exceptions, Tiny-LLM operations return `Result\<T\>` which either contains a value of type `T` or an error message.

## Class Definition

```cpp
namespace tinyllm {

template <typename T>
class Result {
public:
    // Constructors
    Result(T value);                    // Success result
    Result(Error error);                // Error result
    Result(std::string error_message);  // Error from string

    // Check status
    bool ok() const;
    explicit operator bool() const;

    // Access value (undefined behavior if error)
    T& value();
    const T& value() const;
    T& operator*();
    const T& operator*() const;
    T* operator->();
    const T* operator->() const;

    // Access value with default
    T value_or(T default_value) const;

    // Access error
    const Error& error() const;
    std::string error_message() const;

private:
    std::variant<T, Error> data_;
};

// Specialization for void
template <>
class Result<void> {
public:
    Result();                           // Success
    Result(Error error);                // Error

    bool ok() const;
    const Error& error() const;
    std::string error_message() const;
};

}  // namespace tinyllm
```

## Usage Patterns

### Basic Error Handling

```cpp
Result<ModelConfig> config_result = load_config("config.json");
if (!config_result.ok()) {
    std::cerr << "Error: " << config_result.error_message() << std::endl;
    return 1;
}

ModelConfig config = config_result.value();
```

### Chaining Operations

```cpp
Result<void> run_inference(const std::string& model_path, const std::string& prompt) {
    InferenceEngine engine;

    Result<void> load_result = engine.load(model_path);
    if (!load_result.ok()) {
        return load_result;  // Propagate error
    }

    Result<std::string> gen_result = engine.generate(prompt);
    if (!gen_result.ok()) {
        return gen_result.error();
    }

    std::cout << gen_result.value() << std::endl;
    return {};  // Success
}
```

### Using value_or

```cpp
Result\<int\> get_num_layers() {
    // ...
}

int layers = get_num_layers().value_or(32);  // Default to 32 if error
```

### Boolean Conversion

```cpp
Result<void> result = engine.load("model.bin");

if (result) {
    // Success
    std::cout << "Model loaded successfully" << std::endl;
} else {
    // Error
    std::cerr << result.error_message() << std::endl;
}
```

## Error Class

```cpp
namespace tinyllm {

class Error {
public:
    Error(ErrorCode code, std::string message);
    Error(std::string message);  // Generic error

    ErrorCode code() const;
    const std::string& message() const;
    std::string full_message() const;

private:
    ErrorCode code_;
    std::string message_;
};

enum class ErrorCode {
    Success = 0,
    InvalidArgument,
    FileNotFound,
    InvalidFormat,
    OutOfMemory,
    CudaError,
    InternalError,
};

}  // namespace tinyllm
```

## Best Practices

### 1. Always Check Results

```cpp
// BAD: Undefined behavior if error
auto value = result.value();

// GOOD: Check first
if (result.ok()) {
    auto value = result.value();
}
```

### 2. Propagate Errors

```cpp
Result<void> process() {
    Result<Data> data = load_data();
    if (!data.ok()) {
        return data.error();  // Propagate error
    }

    Result<void> processed = transform(data.value());
    if (!processed.ok()) {
        return processed;  // Propagate error
    }

    return {};
}
```

### 3. Use Meaningful Error Messages

```cpp
return Result<ModelConfig>::Error(
    ErrorCode::InvalidArgument,
    "vocab_size must be positive, got: " + std::to_string(vocab_size)
);
```

## Comparison with Exceptions

| Aspect | Result\<T\> | Exceptions |
|--------|-----------|------------|
| Performance | Zero overhead | Stack unwinding cost |
| Control flow | Explicit | Implicit |
| Debuggability | Easy to trace | Can be hard to follow |
| CUDA compatibility | Excellent | Problematic |

## See Also

- [InferenceEngine](/en/api/inference-engine) - Main API class
- [Error Handling Guide](/en/guide/architecture#error-handling)
