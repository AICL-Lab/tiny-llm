// Q5_0 / Q4_K / Q6_K 反量化的单元测试。
//
// 期望值的来源：
// 1. 合成块测试的期望值由 Python `gguf` 包的参考实现
//    （gguf.quants.Q5_0/Q4_K/Q6_K.dequantize）对相同字节序列计算得出；
// 2. 真实模型测试（环境变量 TLLM_GGUF_TEST_MODEL 门控）的期望值由同一参考
//    实现对 Qwen2.5-0.5B-Instruct Q4_K_M 文件的首块反量化结果得出。
// 布局依据见 ggml/gguf 官方格式，不依赖本仓库自身实现计算期望值。

#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/quantization.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace tiny_llm;

namespace {

std::vector<float> toFloats(const std::vector<half> &hs) {
    std::vector<float> out(hs.size());
    for (size_t i = 0; i < hs.size(); ++i) {
        out[i] = __half2float(hs[i]);
    }
    return out;
}

void expectNearF16(float actual, float expected, const char *ctx) {
    const float tol = std::abs(expected) * 1e-3f + 1e-6f;
    EXPECT_NEAR(actual, expected, tol) << ctx;
}

} // namespace

TEST(DequantizeQ5_0, NullPointerFails) {
    auto r = dequantizeQ5_0(nullptr, 1);
    EXPECT_TRUE(r.isErr());
}

TEST(DequantizeQ5_0, MatchesReferenceOnSyntheticBlock) {
    // 合成块: d=2.0, qh=全 1（所有第 5 位置位）, qs=0x5A x 16
    // 值 0..15: 低 nibble 0xA=10 | (1<<4) = 26 -> (26-16)*2 = 20
    // 值 16..31: 高 nibble 0x5 | (1<<4) = 21 -> (21-16)*2 = 10
    uint8_t block[22];
    half    d = __float2half(2.0f);
    std::memcpy(block, &d, sizeof(d));
    uint32_t qh = 0xFFFFFFFFu;
    std::memcpy(block + 2, &qh, sizeof(qh));
    std::memset(block + 6, 0x5A, 16);

    auto r = dequantizeQ5_0(block, 1);
    ASSERT_FALSE(r.isErr()) << r.error();
    auto v = toFloats(r.value());
    ASSERT_EQ(v.size(), 32u);
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(v[i], 20.0f) << "index " << i;
    }
    for (int i = 16; i < 32; ++i) {
        EXPECT_FLOAT_EQ(v[i], 10.0f) << "index " << i;
    }
}

TEST(DequantizeQ4_K, NullPointerFails) {
    auto r = dequantizeQ4_K(nullptr, 1);
    EXPECT_TRUE(r.isErr());
}

TEST(DequantizeQ4_K, MatchesReferenceOnSyntheticBlock) {
    // 合成块: d=1.0, dmin=0.5, scales=[0..11], qs=[0..127]
    uint8_t block[144];
    half    d    = __float2half(1.0f);
    half    dmin = __float2half(0.5f);
    std::memcpy(block, &d, sizeof(d));
    std::memcpy(block + 2, &dmin, sizeof(dmin));
    for (int i = 0; i < 12; ++i) block[4 + i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 128; ++i) block[16 + i] = static_cast<uint8_t>(i);

    auto r = dequantizeQ4_K(block, 1);
    ASSERT_FALSE(r.isErr()) << r.error();
    auto v = toFloats(r.value());
    ASSERT_EQ(v.size(), 256u);

    // Python gguf 参考实现输出
    for (int i = 0; i < 8; ++i) expectNearF16(v[i], -2.0f, "sub-block 0");
    for (int i = 32; i < 40; ++i) expectNearF16(v[i], -2.5f, "sub-block 1");
    const float sb2[8] = {-3.0f, -1.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f};
    for (int i = 0; i < 8; ++i) expectNearF16(v[64 + i], sb2[i], "sub-block 2");
    for (int i = 248; i < 256; ++i) expectNearF16(v[i], 77.0f, "sub-block 7");
}

TEST(DequantizeQ6_K, NullPointerFails) {
    auto r = dequantizeQ6_K(nullptr, 1);
    EXPECT_TRUE(r.isErr());
}

TEST(DequantizeQ6_K, MatchesReferenceOnSyntheticBlock) {
    // 合成块: ql=[0..127], qh=[0..63], scales=[-8..7], d=0.25
    uint8_t block[210];
    for (int i = 0; i < 128; ++i) block[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 64; ++i) block[128 + i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 16; ++i) block[192 + i] = static_cast<uint8_t>(static_cast<int8_t>(i - 8));
    half d = __float2half(0.25f);
    std::memcpy(block + 208, &d, sizeof(d));

    auto r = dequantizeQ6_K(block, 1);
    ASSERT_FALSE(r.isErr()) << r.error();
    auto v = toFloats(r.value());
    ASSERT_EQ(v.size(), 256u);

    // Python gguf 参考实现输出
    const float first8[8] = {64.0f, 30.0f, -4.0f, -38.0f, 56.0f, 22.0f, -12.0f, -46.0f};
    for (int i = 0; i < 8; ++i) expectNearF16(v[i], first8[i], "first 8");
    const float at16[8] = {56.0f, 26.25f, -3.5f, -33.25f, 49.0f, 19.25f, -10.5f, -40.25f};
    for (int i = 0; i < 8; ++i) expectNearF16(v[16 + i], at16[i], "[16:24]");
    for (int i = 128; i < 136; ++i) expectNearF16(v[i], 0.0f, "[128:136]");
    for (int i = 248; i < 256; ++i) expectNearF16(v[i], -43.75f, "last 8");
}

// ── 真实模型验证（门控）──────────────────────────────────────────
// 设置 TLLM_GGUF_TEST_MODEL 指向 Qwen2.5-0.5B-Instruct Q4_K_M GGUF 运行。
// 期望值由 Python gguf 参考实现计算，见文件头注释。

class GGUFRealModelTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const char *path = std::getenv("TLLM_GGUF_TEST_MODEL");
        if (path == nullptr) {
            GTEST_SKIP() << "set TLLM_GGUF_TEST_MODEL to a GGUF file to enable";
        }
        path_ = path;
        parser_ = std::make_unique<GGUFParser>(path_);
        auto r = parser_->parse();
        ASSERT_FALSE(r.isErr()) << r.error();
    }

    std::vector<half> dequantFirstBlock(const std::string &name, GGMLType type, size_t block_bytes,
                                        size_t block_vals) {
        const auto *tensor = parser_->getTensorByName(name);
        EXPECT_NE(tensor, nullptr) << name;
        if (!tensor) return {};
        EXPECT_EQ(static_cast<uint32_t>(tensor->type), static_cast<uint32_t>(type)) << name;
        auto raw = parser_->readTensorData(*tensor);
        EXPECT_FALSE(raw.isErr()) << raw.error();
        if (raw.isErr()) return {};
        switch (type) {
        case GGMLType::Q5_0: return dequantizeQ5_0(raw.value().data(), 1).value();
        case GGMLType::Q4_K: return dequantizeQ4_K(raw.value().data(), 1).value();
        case GGMLType::Q6_K: return dequantizeQ6_K(raw.value().data(), 1).value();
        default: ADD_FAILURE() << "unexpected type"; return {};
        }
    }

    std::string              path_;
    std::unique_ptr<GGUFParser> parser_;
};

TEST_F(GGUFRealModelTest, ConfigMatchesQwen25HalfB) {
    auto cr = parser_->extractModelConfig();
    ASSERT_FALSE(cr.isErr()) << cr.error();
    const auto &cfg = cr.value();

    // 仅当确实是 Qwen2.5-0.5B 时断言精确值（允许该门控测试用于其他模型做冒烟检查）
    if (cfg.hidden_dim == 896 && cfg.num_layers == 24) {
        EXPECT_EQ(cfg.num_heads, 14);
        EXPECT_EQ(cfg.num_kv_heads, 2);
        EXPECT_EQ(cfg.vocab_size, 151936);
        EXPECT_EQ(cfg.intermediate_dim, 4864);
        EXPECT_EQ(cfg.max_seq_len, 32768);
        EXPECT_NEAR(cfg.rope_theta, 1000000.0f, 1.0f);
        EXPECT_NEAR(cfg.rms_norm_eps, 1e-6f, 1e-9f);
        EXPECT_EQ(cfg.eos_token_id, 151645);
    } else {
        EXPECT_GT(cfg.hidden_dim, 0);
        EXPECT_GT(cfg.vocab_size, 0);
    }
}

TEST_F(GGUFRealModelTest, FirstBlocksMatchPythonReference) {
    const auto &cfg = parser_->extractModelConfig().value();
    if (!(cfg.hidden_dim == 896 && cfg.num_layers == 24)) {
        GTEST_SKIP() << "reference values are for Qwen2.5-0.5B-Instruct Q4_K_M";
    }

    {
        auto v = toFloats(dequantFirstBlock("token_embd.weight", GGMLType::Q5_0, 22, 32));
        ASSERT_EQ(v.size(), 32u);
        const float exp8[8] = {-0.0101929f, 0.0407715f, 0.0101929f, -0.0f,
                               -0.0280304f, -0.00254822f, -0.0f, -0.0203857f};
        for (int i = 0; i < 8; ++i) expectNearF16(v[i], exp8[i], "token_embd Q5_0");
    }
    {
        auto v = toFloats(dequantFirstBlock("blk.0.ffn_down.weight", GGMLType::Q6_K, 210, 256));
        ASSERT_EQ(v.size(), 256u);
        const float exp8[8] = {-0.00611287f, 0.00611287f, 0.0279446f, 0.00436634f,
                               -0.00523961f, -0.00698614f, -0.0183386f, 0.0026198f};
        for (int i = 0; i < 8; ++i) expectNearF16(v[i], exp8[i], "ffn_down Q6_K first8");
        const float exp128[4] = {0.00285149f, -0.00142574f, -0.00142574f, 0.0456238f};
        for (int i = 0; i < 4; ++i) expectNearF16(v[128 + i], exp128[i], "ffn_down Q6_K [128:132]");
    }
    {
        auto v = toFloats(dequantFirstBlock("blk.23.ffn_down.weight", GGMLType::Q4_K, 144, 256));
        ASSERT_EQ(v.size(), 256u);
        const float exp8[8] = {0.00669551f, 0.00373709f, 0.00669551f, 0.00373709f,
                               -0.00513816f, 0.000778675f, -0.011055f, 0.000778675f};
        for (int i = 0; i < 8; ++i) expectNearF16(v[i], exp8[i], "blk.23 Q4_K first8");
        const float exp128[4] = {0.00363874f, 0.0179163f, -0.00825924f, 0.00601834f};
        for (int i = 0; i < 4; ++i) expectNearF16(v[128 + i], exp128[i], "blk.23 Q4_K [128:132]");
    }
}
