#pragma once

#include <stdexcept>
#include <string>
#include <variant>

namespace tiny_llm {

// Result type for error propagation without exceptions
template <typename T>
class Result {
  public:
    // Factory methods
    static Result<T> ok(T value) { return Result(std::move(value)); }

    static Result<T> err(std::string message) { return Result(Error{std::move(message)}); }

    // Status checks
    bool isOk() const { return std::holds_alternative<T>(data_); }

    bool isErr() const { return !isOk(); }

    // Value access (throws if error)
    T &value() {
        if (isErr()) {
            throw std::runtime_error("Attempted to access value of error Result: " + error());
        }
        return std::get<T>(data_);
    }

    const T &value() const {
        if (isErr()) {
            throw std::runtime_error("Attempted to access value of error Result: " + error());
        }
        return std::get<T>(data_);
    }

    // Error access (throws if ok)
    const std::string &error() const {
        if (isOk()) {
            throw std::runtime_error("Attempted to access error of ok Result");
        }
        return std::get<Error>(data_).message;
    }

    // Value access with default
    T valueOr(T default_value) const {
        if (isOk()) {
            return std::get<T>(data_);
        }
        return default_value;
    }

    // Monadic operations
    template <typename F>
    auto map(F &&f) -> Result<decltype(f(std::declval<T>()))> {
        using U = decltype(f(std::declval<T>()));
        if (isOk()) {
            return Result<U>::ok(f(value()));
        }
        return Result<U>::err(error());
    }

    template <typename F>
    auto flatMap(F &&f) -> decltype(f(std::declval<T>())) {
        if (isOk()) {
            return f(value());
        }
        using U = typename decltype(f(std::declval<T>()))::ValueType;
        return Result<U>::err(error());
    }

    // Type alias for flatMap
    using ValueType = T;

  private:
    struct Error {
        std::string message;
    };

    explicit Result(T value) : data_(std::move(value)) {}
    explicit Result(Error error) : data_(std::move(error)) {}

    std::variant<T, Error> data_;
};

// Specialization for void
template <>
class Result<void> {
  public:
    static Result<void> ok() { return Result(true); }

    static Result<void> err(std::string message) { return Result(std::move(message)); }

    bool isOk() const { return is_ok_; }
    bool isErr() const { return !is_ok_; }

    const std::string &error() const {
        if (isOk()) {
            throw std::runtime_error("Attempted to access error of ok Result");
        }
        return error_;
    }

  private:
    explicit Result(bool) : is_ok_(true) {}
    explicit Result(std::string error) : is_ok_(false), error_(std::move(error)) {}

    bool        is_ok_;
    std::string error_;
};

} // namespace tiny_llm
