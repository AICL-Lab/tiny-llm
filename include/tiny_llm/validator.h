#pragma once

#include "tiny_llm/result.h"
#include "tiny_llm/types.h"

#include <sstream>
#include <string>
#include <vector>

namespace tiny_llm {

/**
 * @brief Validation utility for input checking
 *
 * Provides static methods for validating various input types:
 * - Token IDs and sequences
 * - Generation configuration
 * - Position and sequence length
 * - Pointers and other values
 *
 * Usage:
 * @code
 * auto result = Validator::validateTokenId(token, vocab_size, "generate");
 * if (result.isErr()) {
 *   return Result<vector<int>>::err(result.error());
 * }
 * @endcode
 */
class Validator {
public:
  // ── Token Validation ─────────────────────────────────────────────

  /**
   * @brief Validate a single token ID
   * @param token_id Token ID to validate
   * @param vocab_size Vocabulary size (valid range: 0 to vocab_size-1)
   * @param context Optional context string for error messages
   * @return Result<void> indicating success or error with message
   */
  static Result<void> validateTokenId(int token_id, int vocab_size,
                                       const std::string &context = "") {
    if (vocab_size <= 0) {
      return Result<void>::err(
          buildContext(context, "Invalid vocab_size: " +
                                     std::to_string(vocab_size) +
                                     " (must be positive)"));
    }
    if (token_id < 0 || token_id >= vocab_size) {
      return Result<void>::err(buildContext(
          context, "Invalid token ID: " + std::to_string(token_id) +
                       " (valid range: 0-" + std::to_string(vocab_size - 1) +
                       ")"));
    }
    return Result<void>::ok();
  }

  /**
   * @brief Validate a sequence of token IDs
   * @param tokens Vector of token IDs to validate
   * @param vocab_size Vocabulary size
   * @param context Optional context string for error messages
   * @return Result<void> indicating success or error with message
   */
  static Result<void> validateTokenSequence(const std::vector<int> &tokens,
                                             int vocab_size,
                                             const std::string &context = "") {
    if (tokens.empty()) {
      return Result<void>::err(buildContext(context, "Token sequence is empty"));
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
      auto result = validateTokenId(
          tokens[i], vocab_size,
          context + " token[" + std::to_string(i) + "]");
      if (result.isErr()) {
        return result;
      }
    }
    return Result<void>::ok();
  }

  // ── Configuration Validation ──────────────────────────────────────

  /**
   * @brief Validate generation configuration
   * @param config Generation configuration to validate
   * @return Result<void> indicating success or error with message
   */
  static Result<void>
  validateGenerationConfig(const GenerationConfig &config) {
    if (config.max_new_tokens <= 0) {
      return Result<void>::err(
          "max_new_tokens must be positive: " +
          std::to_string(config.max_new_tokens));
    }
    if (config.temperature <= 0.0f) {
      return Result<void>::err("temperature must be positive: " +
                                std::to_string(config.temperature));
    }
    if (config.top_k < 0) {
      return Result<void>::err("top_k must be non-negative: " +
                                std::to_string(config.top_k));
    }
    if (config.top_p <= 0.0f || config.top_p > 1.0f) {
      return Result<void>::err("top_p must be in (0, 1]: " +
                                std::to_string(config.top_p));
    }
    if (config.repetition_penalty <= 0.0f) {
      return Result<void>::err("repetition_penalty must be positive: " +
                                std::to_string(config.repetition_penalty));
    }
    return Result<void>::ok();
  }

  /**
   * @brief Validate model configuration
   * @param config Model configuration to validate
   * @return Result<void> indicating success or error with message
   */
  static Result<void> validateModelConfig(const ModelConfig &config) {
    if (config.vocab_size <= 0) {
      return Result<void>::err("vocab_size must be positive: " +
                                std::to_string(config.vocab_size));
    }
    if (config.hidden_dim <= 0) {
      return Result<void>::err("hidden_dim must be positive: " +
                                std::to_string(config.hidden_dim));
    }
    if (config.num_layers <= 0) {
      return Result<void>::err("num_layers must be positive: " +
                                std::to_string(config.num_layers));
    }
    if (config.num_heads <= 0) {
      return Result<void>::err("num_heads must be positive: " +
                                std::to_string(config.num_heads));
    }
    if (config.num_kv_heads <= 0) {
      return Result<void>::err("num_kv_heads must be positive: " +
                                std::to_string(config.num_kv_heads));
    }
    if (config.head_dim <= 0) {
      return Result<void>::err("head_dim must be positive: " +
                                std::to_string(config.head_dim));
    }
    if (config.max_seq_len <= 0) {
      return Result<void>::err("max_seq_len must be positive: " +
                                std::to_string(config.max_seq_len));
    }
    if (config.rms_norm_eps <= 0.0f) {
      return Result<void>::err("rms_norm_eps must be positive: " +
                                std::to_string(config.rms_norm_eps));
    }
    return Result<void>::ok();
  }

  // ── Position/Length Validation ────────────────────────────────────

  /**
   * @brief Validate position index
   * @param position Position index to validate
   * @param max_seq_len Maximum sequence length
   * @param context Optional context string for error messages
   * @return Result<void> indicating success or error with message
   */
  static Result<void> validatePosition(int position, int max_seq_len,
                                        const std::string &context = "") {
    if (position < 0 || position >= max_seq_len) {
      return Result<void>::err(
          buildContext(context, "Invalid position: " +
                                     std::to_string(position) +
                                     " (valid range: 0-" +
                                     std::to_string(max_seq_len - 1) + ")"));
    }
    return Result<void>::ok();
  }

  /**
   * @brief Validate sequence length
   * @param seq_len Sequence length to validate
   * @param max_seq_len Maximum sequence length
   * @param context Optional context string for error messages
   * @return Result<void> indicating success or error with message
   */
  static Result<void> validateSeqLen(int seq_len, int max_seq_len,
                                      const std::string &context = "") {
    if (seq_len <= 0) {
      return Result<void>::err(
          buildContext(context, "seq_len must be positive: " +
                                     std::to_string(seq_len)));
    }
    if (seq_len > max_seq_len) {
      return Result<void>::err(
          buildContext(context, "seq_len " + std::to_string(seq_len) +
                                     " exceeds max_seq_len " +
                                     std::to_string(max_seq_len)));
    }
    return Result<void>::ok();
  }

  /**
   * @brief Validate that prompt length doesn't exceed max_seq_len
   * @param prompt_len Prompt length
   * @param max_new_tokens Maximum new tokens to generate
   * @param max_seq_len Maximum sequence length
   * @return Result<void> indicating success or error with message
   */
  static Result<void> validatePromptLength(int prompt_len, int max_new_tokens,
                                            int max_seq_len) {
    if (prompt_len <= 0) {
      return Result<void>::err("Prompt length must be positive: " +
                                std::to_string(prompt_len));
    }
    if (prompt_len >= max_seq_len) {
      return Result<void>::err(
          "Prompt length " + std::to_string(prompt_len) +
          " exceeds max_seq_len " + std::to_string(max_seq_len));
    }
    int total_len = prompt_len + max_new_tokens;
    if (total_len > max_seq_len) {
      return Result<void>::err("Total length (prompt + max_new_tokens = " +
                                std::to_string(total_len) +
                                ") exceeds max_seq_len " +
                                std::to_string(max_seq_len));
    }
    return Result<void>::ok();
  }

  // ── Pointer Validation ────────────────────────────────────────────

  /**
   * @brief Validate that a pointer is not null
   * @tparam T Pointer type
   * @param ptr Pointer to validate
   * @param name Name of the pointer for error message
   * @return Result<void> indicating success or error with message
   */
  template <typename T>
  static Result<void> validateNotNull(const T *ptr,
                                       const std::string &name) {
    if (ptr == nullptr) {
      return Result<void>::err(name + " is null");
    }
    return Result<void>::ok();
  }

  // ── Range Validation ──────────────────────────────────────────────

  /**
   * @brief Validate that a value is within a range
   * @param value Value to validate
   * @param min Minimum value (inclusive)
   * @param max Maximum value (inclusive)
   * @param name Name for error message
   * @return Result<void> indicating success or error with message
   */
  static Result<void> validateRange(int value, int min, int max,
                                     const std::string &name) {
    if (value < min || value > max) {
      return Result<void>::err(name + " value " + std::to_string(value) +
                                " out of range [" + std::to_string(min) +
                                ", " + std::to_string(max) + "]");
    }
    return Result<void>::ok();
  }

  /**
   * @brief Validate that a float value is within a range
   * @param value Value to validate
   * @param min Minimum value (inclusive)
   * @param max Maximum value (inclusive)
   * @param name Name for error message
   * @return Result<void> indicating success or error with message
   */
  static Result<void> validateRange(float value, float min, float max,
                                     const std::string &name) {
    if (value < min || value > max) {
      return Result<void>::err(name + " value " + std::to_string(value) +
                                " out of range [" + std::to_string(min) +
                                ", " + std::to_string(max) + "]");
    }
    return Result<void>::ok();
  }

  // ── Layer Index Validation ────────────────────────────────────────

  /**
   * @brief Validate layer index
   * @param layer_idx Layer index to validate
   * @param num_layers Total number of layers
   * @param context Optional context string for error messages
   * @return Result<void> indicating success or error with message
   */
  static Result<void> validateLayerIndex(int layer_idx, int num_layers,
                                          const std::string &context = "") {
    if (layer_idx < 0 || layer_idx >= num_layers) {
      return Result<void>::err(
          buildContext(context, "Invalid layer_idx: " +
                                     std::to_string(layer_idx) +
                                     " (valid range: 0-" +
                                     std::to_string(num_layers - 1) + ")"));
    }
    return Result<void>::ok();
  }

private:
  /**
   * @brief Build context-prefixed error message
   * @param context Context prefix
   * @param message Error message
   * @return Combined error message
   */
  static std::string buildContext(const std::string &context,
                                   const std::string &message) {
    if (context.empty()) {
      return message;
    }
    return context + ": " + message;
  }
};

// ── Convenience Macros ────────────────────────────────────────────────

/**
 * @brief Validate condition and return error if false
 * @param condition Condition to validate
 * @param message Error message if condition is false
 */
#define TLLM_VALIDATE(condition, message)                                       \
  do {                                                                          \
    if (!(condition)) {                                                         \
      return tiny_llm::Result<void>::err(message);                              \
    }                                                                           \
  } while (0)

/**
 * @brief Validate and propagate error if any
 * @param result Result to check
 */
#define TLLM_VALIDATE_RESULT(result)                                            \
  do {                                                                          \
    if ((result).isErr()) {                                                     \
      return tiny_llm::Result<void>::err((result).error());                     \
    }                                                                           \
  } while (0)

} // namespace tiny_llm
