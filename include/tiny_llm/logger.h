#pragma once

#include <memory>
#include <string>

// Forward declare spdlog types to avoid header pollution
namespace spdlog {
class logger;
namespace sinks {
class sink;
}
} // namespace spdlog

namespace tiny_llm {

/**
 * @brief Log level enumeration
 */
enum class LogLevel { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, CRITICAL = 5, OFF = 6 };

/**
 * @brief Logger singleton for tiny-llm
 *
 * Provides a centralized logging system with support for:
 * - Console output (colorized)
 * - File output (rotating files)
 * - Async logging for performance
 * - Multiple log levels
 *
 * Usage:
 * @code
 * // Initialize at program start
 * tiny_llm::Logger::init(tiny_llm::LogLevel::INFO, "tiny_llm.log", false);
 *
 * // Log messages
 * TLLM_INFO("Loading model from: {}", model_path);
 * TLLM_DEBUG("Config: hidden_dim={}", config.hidden_dim);
 * TLLM_ERROR("Failed to load: {}", error_msg);
 *
 * // Shutdown at program end
 * tiny_llm::Logger::shutdown();
 * @endcode
 */
class Logger {
  public:
    /**
     * @brief Initialize the logging system
     * @param level Minimum log level to output
     * @param log_file Optional path for file logging (empty = console only)
     * @param async Enable asynchronous logging for better performance
     */
    static void init(LogLevel level = LogLevel::INFO, const std::string &log_file = "",
                     bool async = false);

    /**
     * @brief Shutdown the logging system
     *
     * Should be called before program exit to flush all logs.
     */
    static void shutdown();

    /**
     * @brief Get the underlying spdlog logger
     * @return Shared pointer to the logger instance
     *
     * If not initialized, initializes with default settings.
     */
    static std::shared_ptr<spdlog::logger> get();

    /**
     * @brief Set the log level
     * @param level New minimum log level
     */
    static void setLevel(LogLevel level);

    /**
     * @brief Flush all pending log messages
     */
    static void flush();

    /**
     * @brief Check if logger is initialized
     * @return true if initialized, false otherwise
     */
    static bool isInitialized() { return initialized_; }

  private:
    static std::shared_ptr<spdlog::logger> logger_;
    static bool                            initialized_;
};

} // namespace tiny_llm

// ── Log Macros ───────────────────────────────────────────────────────

// Include spdlog header for macros
#include <spdlog/spdlog.h>

#define TLLM_TRACE(...)    SPDLOG_LOGGER_TRACE(tiny_llm::Logger::get(), __VA_ARGS__)
#define TLLM_DEBUG(...)    SPDLOG_LOGGER_DEBUG(tiny_llm::Logger::get(), __VA_ARGS__)
#define TLLM_INFO(...)     SPDLOG_LOGGER_INFO(tiny_llm::Logger::get(), __VA_ARGS__)
#define TLLM_WARN(...)     SPDLOG_LOGGER_WARN(tiny_llm::Logger::get(), __VA_ARGS__)
#define TLLM_ERROR(...)    SPDLOG_LOGGER_ERROR(tiny_llm::Logger::get(), __VA_ARGS__)
#define TLLM_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(tiny_llm::Logger::get(), __VA_ARGS__)

// Conditional logging (for performance-critical paths)
#define TLLM_TRACE_IF(cond, ...)                                                                   \
    if (tiny_llm::Logger::get()->should_log(spdlog::level::trace)) {                               \
        TLLM_TRACE(__VA_ARGS__);                                                                   \
    }
#define TLLM_DEBUG_IF(cond, ...)                                                                   \
    if (tiny_llm::Logger::get()->should_log(spdlog::level::debug)) {                               \
        TLLM_DEBUG(__VA_ARGS__);                                                                   \
    }
