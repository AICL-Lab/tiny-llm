#include "tiny_llm/logger.h"

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <mutex>
#include <vector>

namespace tiny_llm {

std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;
std::atomic<bool>               Logger::initialized_{false};

namespace {
std::mutex g_logger_mutex;
}

// 实际初始化（调用方须持有 g_logger_mutex）。内部不使用 TLLM_* 宏——
// 它们会调用 get() 再次加锁，造成锁内重入死锁。
void Logger::initLocked(LogLevel level, const std::string &log_file, bool async) {

    std::vector<spdlog::sink_ptr> sinks;

    // Console sink (colorized)
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
    sinks.push_back(console_sink);

    // File sink (optional, rotating)
    if (!log_file.empty()) {
        // 10MB per file, keep 3 rotated files
        auto file_sink =
            std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_file, 1024 * 1024 * 10, 3);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] [%s:%#] %v");
        sinks.push_back(file_sink);
    }

    // Create logger (async or sync)
    if (async) {
        // Initialize thread pool with 8192 queue size and 1 worker thread
        spdlog::init_thread_pool(8192, 1);
        logger_ = std::make_shared<spdlog::async_logger>("tiny_llm", sinks.begin(), sinks.end(),
                                                         spdlog::thread_pool(),
                                                         spdlog::async_overflow_policy::block);
    } else {
        logger_ = std::make_shared<spdlog::logger>("tiny_llm", sinks.begin(), sinks.end());
    }

    // Set log level
    logger_->set_level(static_cast<spdlog::level::level_enum>(level));

    // Auto-flush on warnings and above
    logger_->flush_on(spdlog::level::warn);

    // Register as default logger
    spdlog::register_logger(logger_);
    spdlog::set_default_logger(logger_);

    if (logger_) {
        logger_->debug("Logger initialized: level={}, async={}, file={}", static_cast<int>(level),
                       async, log_file.empty() ? "(console only)" : log_file);
    }
}

void Logger::init(LogLevel level, const std::string &log_file, bool async) {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    if (initialized_.load(std::memory_order_relaxed)) {
        return;
    }
    initLocked(level, log_file, async);
    initialized_.store(true, std::memory_order_release);
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    if (!initialized_.load(std::memory_order_relaxed)) {
        return;
    }
    if (logger_) {
        logger_->debug("Logger shutting down");
    }
    spdlog::shutdown();
    logger_ = nullptr;
    initialized_.store(false, std::memory_order_release);
}

std::shared_ptr<spdlog::logger> Logger::get() {
    if (!initialized_.load(std::memory_order_acquire)) {
        // double-checked locking：未初始化时加锁补齐
        std::lock_guard<std::mutex> lock(g_logger_mutex);
        if (!initialized_.load(std::memory_order_relaxed)) {
            initLocked(LogLevel::INFO, "", false);
            initialized_.store(true, std::memory_order_release);
        }
    }
    return logger_;
}

void Logger::setLevel(LogLevel level) {
    if (logger_) {
        logger_->set_level(static_cast<spdlog::level::level_enum>(level));
        TLLM_DEBUG("Log level changed to {}", static_cast<int>(level));
    }
}

void Logger::flush() {
    if (logger_) {
        logger_->flush();
    }
}

bool Logger::isInitialized() { return initialized_.load(std::memory_order_acquire); }

} // namespace tiny_llm
