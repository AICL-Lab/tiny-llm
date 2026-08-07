#pragma once

#include "tiny_llm/result.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

/**
 * @brief GGUF 内嵌 tokenizer 的原始数据（tokenizer.ggml.* 元数据）
 */
struct TokenizerData {
    std::string              model_type; // "gpt2"（BPE）/ "llama"（SentencePiece）等
    std::vector<std::string> tokens;     // id -> piece（gpt2 风格字节编码的 UTF-8）
    std::vector<std::string> merges;     // rank -> "piece1 piece2"（gpt2 BPE 合并规则）
    std::vector<int>         token_types; // 每 token 的 GGML 类型（1=NORMAL 3=CONTROL 4=USER_DEFINED 5=UNUSED）
    int                      bos_token_id = -1;
    int                      eos_token_id = -1;
    int                      padding_token_id = -1;
    bool                     add_bos_token = false;
};

/**
 * @brief gpt2 风格字节级 BPE tokenizer（Qwen2/GPT 系 GGUF 词表）
 *
 * 流水线与 HuggingFace tokenizers 库一致：
 *   1. 特殊 token 精确匹配隔离（added tokens）
 *   2. Qwen2 正则预分词（Unicode 感知，见 preTokenize）
 *   3. GPT-2 字节编码（byte -> unicode 映射）
 *   4. 按合并规则优先级的 BPE
 *
 * 正确性由 tests/test_tokenizer.cpp 对照 HF tokenizers 库的差分测试保证。
 */
class Tokenizer {
  public:
    /**
     * @brief 从 TokenizerData 构建（当前仅支持 gpt2 BPE 类型）
     */
    static Result<Tokenizer> build(const TokenizerData &data);

    /**
     * @brief 文本 -> token id 序列
     * @param text UTF-8 文本
     * @param add_bos 是否在开头添加 bos token
     */
    std::vector<int> encode(const std::string &text, bool add_bos = false) const;

    /**
     * @brief token id 序列 -> UTF-8 文本（字节级解码，还原 Ġ 等映射字符）
     */
    std::string decode(const std::vector<int> &ids) const;

    int  vocabSize() const { return static_cast<int>(id_to_token_.size()); }
    int  bosTokenId() const { return bos_id_; }
    int  eosTokenId() const { return eos_id_; }
    bool pieceIsSpecial(int id) const;

  private:
    std::vector<std::string> preTokenize(const std::string &text) const;
    std::vector<std::string> bpe(const std::string &word) const;

    std::vector<std::string>                        id_to_token_;
    std::unordered_map<std::string, int>            token_to_id_;
    std::unordered_map<std::string, int>            merge_rank_; // "a b" -> rank
    std::vector<std::pair<std::string, int>>        special_tokens_; // 按长度降序
    int                                             bos_id_ = -1;
    int                                             eos_id_ = -1;
};

struct GGUFMetadata;

/**
 * @brief 从 GGUF 元数据（tokenizer.ggml.*）提取 TokenizerData
 */
Result<TokenizerData> loadTokenizerData(const GGUFMetadata &md);

// GPT-2 字节编码表：byte -> unicode 码点（可打印区间为自身，其余映射到 256+n）
const uint32_t *byteToUnicodeTable();
// 逆映射：unicode 码点 -> byte，无效返回 -1
int unicodeToByte(uint32_t cp);

} // namespace tiny_llm
