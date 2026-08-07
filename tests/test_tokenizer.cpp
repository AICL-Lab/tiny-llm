// tokenizer 差分测试：对照 HuggingFace tokenizers 库的权威编码结果。
// 门控用例需 TLLM_GGUF_TEST_MODEL 指向 Qwen2.5-0.5B-Instruct 的 GGUF 文件。
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/tokenizer.h"
#include "tokenizer_fixture_cases.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

using namespace tiny_llm;

namespace {

std::optional<std::string> modelPath() {
    if (const char *p = std::getenv("TLLM_GGUF_TEST_MODEL"); p && *p)
        return std::string(p);
    return std::nullopt;
}

// 把单个码点编码为 UTF-8（测试夹具用）
std::string utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

} // namespace

// ── 无需模型的单元测试 ─────────────────────────────────────────

// GPT-2 字节编码表：可打印字节映射到自身，其余按升序映射到 256+n
TEST(Tokenizer, ByteEncodingTable) {
    const uint32_t *t = byteToUnicodeTable();
    EXPECT_EQ(t['!'], 33u);
    EXPECT_EQ(t['~'], 126u);
    EXPECT_EQ(t[0xA1], 0xA1u); // 161 在可打印区间，映射到自身
    EXPECT_EQ(t[0xFF], 0xFFu);
    EXPECT_EQ(t[0xA0], 322u);  // 160(NBSP) 不在可打印区间 -> 256+66
    EXPECT_EQ(t[0x00], 256u);
    EXPECT_EQ(t[0x01], 257u);
    for (uint32_t b = 0; b < 256; ++b)
        EXPECT_EQ(unicodeToByte(t[b]), static_cast<int>(b));
    EXPECT_EQ(unicodeToByte(0xDEADBEEF), -1);
}

// 最小合成词表验证 encode 流水线（特殊 token 隔离 -> 预分词 -> 字节编码 -> BPE）
TEST(Tokenizer, SyntheticEncode) {
    const uint32_t *t = byteToUnicodeTable();
    TokenizerData d;
    d.model_type = "gpt2";
    d.tokens.resize(256);
    for (uint32_t b = 0; b < 256; ++b)
        d.tokens[b] = utf8(t[b]);
    d.token_types.assign(256, 1);
    const std::string a = utf8(t['a']), b = utf8(t['b']);
    d.merges = {a + " " + b};
    d.tokens.push_back(a + b);
    d.token_types.push_back(1);
    d.tokens.push_back("<|x|>");
    d.token_types.push_back(3); // CONTROL -> 特殊 token
    const int id_ab = 256;
    const int id_sp = 257;

    auto r = Tokenizer::build(d);
    ASSERT_TRUE(r.isOk()) << r.error();
    const Tokenizer &tok = r.value();

    EXPECT_EQ(tok.encode("ab", false), std::vector<int>({id_ab}));
    EXPECT_EQ(tok.encode("ba", false).size(), 2u);
    EXPECT_TRUE(tok.pieceIsSpecial(id_sp));
    EXPECT_FALSE(tok.pieceIsSpecial(id_ab));
    EXPECT_EQ(tok.decode(tok.encode("ab", false)), "ab");
    EXPECT_EQ(tok.decode({id_sp}), "<|x|>");
}

// ── 真实模型差分测试（门控） ───────────────────────────────────

class TokenizerRealModel : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        if (auto p = modelPath())
            g_path = *p;
    }
    static inline std::string g_path;
};

// C++ encode 与 HuggingFace tokenizers 权威输出逐 id 对齐
TEST_F(TokenizerRealModel, DifferentialAgainstHuggingFace) {
    if (g_path.empty())
        GTEST_SKIP() << "set TLLM_GGUF_TEST_MODEL to a Qwen2.5 GGUF to enable";
    GGUFParser parser(g_path);
    ASSERT_TRUE(parser.parse().isOk()) << "failed to parse GGUF";
    auto td = loadTokenizerData(parser.getMetadata());
    ASSERT_TRUE(td.isOk()) << td.error();
    auto tr = Tokenizer::build(td.value());
    ASSERT_TRUE(tr.isOk()) << tr.error();
    const Tokenizer &tok = tr.value();
    EXPECT_EQ(tok.vocabSize(), 151936);

    int failures = 0;
    for (const auto &c : test_fixture::kCases) {
        auto ids = tok.encode(c.text, false);
        if (ids != c.ids) {
            ++failures;
            if (failures <= 5) {
                std::string got, want;
                for (int id : ids)
                    got += std::to_string(id) + " ";
                for (int id : c.ids)
                    want += std::to_string(id) + " ";
                ADD_FAILURE() << "mismatch text=" << c.text << "\n got(" << ids.size()
                              << "): " << got << "\nwant(" << c.ids.size() << "): " << want;
            }
        }
    }
    EXPECT_EQ(failures, 0) << failures << " / " << test_fixture::kCaseCount << " cases mismatched";
}

// decode(encode(text)) == text（字节级 BPE 无损往返）
TEST_F(TokenizerRealModel, RoundTripDecode) {
    if (g_path.empty())
        GTEST_SKIP() << "set TLLM_GGUF_TEST_MODEL to a Qwen2.5 GGUF to enable";
    GGUFParser parser(g_path);
    ASSERT_TRUE(parser.parse().isOk());
    auto td = loadTokenizerData(parser.getMetadata());
    ASSERT_TRUE(td.isOk());
    auto tr = Tokenizer::build(td.value());
    ASSERT_TRUE(tr.isOk());
    const Tokenizer &tok = tr.value();

    for (const auto &c : test_fixture::kCases) {
        auto dec = tok.decode(tok.encode(c.text, false));
        EXPECT_EQ(dec, std::string(c.text)) << "round-trip failed for: " << c.text;
    }
}
