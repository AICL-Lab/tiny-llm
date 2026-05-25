#include "tiny_llm/inference_engine.h"
#include "tiny_llm/model_loader.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <random>
// #include <rapidcheck.h>
// NOTE: rapidcheck/gtest disabled due to GCC 11/12 std::function bug
// in CI builds.
// #include <rapidcheck/gtest.h>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace tiny_llm;

// Helper to create temporary files
class TempFile {
  public:
    TempFile(const std::string &suffix = ".bin") {
#ifdef _WIN32
        int pid = _getpid();
#else
        int pid = getpid();
#endif
        path_ = "/tmp/tiny_llm_test_" + std::to_string(pid) + "_" +
                std::to_string(std::random_device{}()) + suffix;
    }

    ~TempFile() { std::remove(path_.c_str()); }

    const std::string &path() const { return path_; }

    void write(const void *data, size_t size) {
        std::ofstream file(path_, std::ios::binary);
        file.write(reinterpret_cast<const char *>(data), size);
    }

    template <typename T>
    void write(const T &value) {
        write(&value, sizeof(T));
    }

    void writeBytes(const std::vector<uint8_t> &bytes) const {
        std::ofstream file(path_, std::ios::binary);
        file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }

  private:
    std::string path_;
};

namespace {

void appendBytes(std::vector<uint8_t> &out, const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

template <typename T>
void appendValue(std::vector<uint8_t> &out, const T &value) {
    appendBytes(out, &value, sizeof(T));
}

void appendString(std::vector<uint8_t> &out, const std::string &value) {
    uint64_t size = value.size();
    appendValue(out, size);
    appendBytes(out, value.data(), value.size());
}

struct GGUFFileBuilder {
    std::vector<uint8_t> metadata_entries;
    std::vector<uint8_t> tensor_entries;
    uint64_t             metadata_count = 0;
    uint64_t             tensor_count = 0;

    template <typename T>
    void addScalarMetadata(const std::string &key, GGUFType type, const T &value) {
        appendString(metadata_entries, key);
        uint32_t raw_type = static_cast<uint32_t>(type);
        appendValue(metadata_entries, raw_type);
        appendValue(metadata_entries, value);
        ++metadata_count;
    }

    void addArrayMetadataHeader(const std::string &key, GGUFType elem_type, uint64_t count) {
        appendString(metadata_entries, key);
        uint32_t raw_type = static_cast<uint32_t>(GGUFType::ARRAY);
        appendValue(metadata_entries, raw_type);
        uint32_t raw_elem_type = static_cast<uint32_t>(elem_type);
        appendValue(metadata_entries, raw_elem_type);
        appendValue(metadata_entries, count);
        ++metadata_count;
    }

    void writeTo(const TempFile &file) const {
        std::vector<uint8_t> bytes;
        appendValue(bytes, GGUF_MAGIC);
        uint32_t version = 3;
        appendValue(bytes, version);
        appendValue(bytes, tensor_count);
        appendValue(bytes, metadata_count);
        bytes.insert(bytes.end(), metadata_entries.begin(), metadata_entries.end());
        bytes.insert(bytes.end(), tensor_entries.begin(), tensor_entries.end());

        constexpr uint64_t kAlignment = 32;
        size_t             aligned_size = (bytes.size() + kAlignment - 1) & ~(kAlignment - 1);
        bytes.resize(aligned_size, 0);
        file.writeBytes(bytes);
    }
};

} // namespace

// Unit tests for GGUF header parsing
class GGUFHeaderTest : public ::testing::Test {};

TEST(GGUFTensorInfoTest, NumElementsReturnsZeroOnOverflow) {
    GGUFTensorInfo info;
    info.dimensions = {std::numeric_limits<uint64_t>::max(), 2};

    EXPECT_EQ(info.numElements(), 0u);
}

TEST(GGUFTensorInfoTest, CalculateSizeReturnsZeroOnOverflow) {
    GGUFTensorInfo info;
    info.type = GGMLType::F32;
    info.dimensions = {std::numeric_limits<uint64_t>::max(), 2};

    EXPECT_EQ(info.calculateSize(), 0u);
}

TEST_F(GGUFHeaderTest, OversizedArrayReturnsErrorInsteadOfThrowing) {
    TempFile        file(".gguf");
    GGUFFileBuilder gguf;
    gguf.addArrayMetadataHeader("bad.array", GGUFType::UINT32,
                                std::numeric_limits<uint64_t>::max() / sizeof(uint32_t));
    gguf.writeTo(file);

    GGUFParser parser(file.path());

    EXPECT_NO_THROW({
        auto result = parser.parse();
        EXPECT_TRUE(result.isErr());
        if (result.isErr()) {
            EXPECT_NE(result.error().find("Array"), std::string::npos);
        }
    });
}

TEST_F(GGUFHeaderTest, ExtractModelConfigIgnoresNumericArchitectureMetadata) {
    TempFile        file(".gguf");
    GGUFFileBuilder gguf;
    gguf.addScalarMetadata("general.architecture", GGUFType::UINT32, uint32_t{1234});
    gguf.writeTo(file);

    GGUFParser parser(file.path());
    auto       parse_result = parser.parse();
    ASSERT_TRUE(parse_result.isOk()) << parse_result.error();

    auto config_result = parser.extractModelConfig();
    ASSERT_TRUE(config_result.isOk()) << config_result.error();
    EXPECT_EQ(config_result.value().hidden_dim, ModelConfig{}.hidden_dim);
}

TEST_F(GGUFHeaderTest, IncompleteRuntimeGGUFReturnsStructuredErrorWithoutThrowing) {
    TempFile        file(".gguf");
    GGUFFileBuilder gguf;
    gguf.addScalarMetadata("llama.embedding_length", GGUFType::UINT32, uint32_t{8});
    gguf.addScalarMetadata("llama.block_count", GGUFType::UINT32, uint32_t{1});
    gguf.addScalarMetadata("llama.attention.head_count", GGUFType::UINT32, uint32_t{1});
    gguf.addScalarMetadata("llama.attention.head_count_kv", GGUFType::UINT32, uint32_t{1});
    gguf.addScalarMetadata("llama.context_length", GGUFType::UINT32, uint32_t{16});
    gguf.addScalarMetadata("llama.feed_forward_length", GGUFType::UINT32, uint32_t{16});
    gguf.addScalarMetadata("tokenizer.ggml.model.vocab_size", GGUFType::UINT32, uint32_t{32});
    gguf.writeTo(file);

    ModelConfig config;

    EXPECT_NO_THROW({
        auto result = ModelLoader::loadGGUF(file.path(), config);
        EXPECT_TRUE(result.isErr());
        if (result.isErr()) {
            EXPECT_TRUE(result.error().find("missing") != std::string::npos ||
                        result.error().find("required") != std::string::npos ||
                        result.error().find("unsupported") != std::string::npos);
        }
    });
}

TEST_F(GGUFHeaderTest, ValidHeader) {
    TempFile file(".gguf");

    // Write valid GGUF header
    std::vector<uint8_t> header;

    // Magic: "GGUF" = 0x46554747
    uint32_t magic = GGUF_MAGIC;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&magic),
                  reinterpret_cast<uint8_t *>(&magic) + 4);

    // Version: 3
    uint32_t version = 3;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&version),
                  reinterpret_cast<uint8_t *>(&version) + 4);

    // Tensor count: 10
    uint64_t tensor_count = 10;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&tensor_count),
                  reinterpret_cast<uint8_t *>(&tensor_count) + 8);

    // Metadata KV count: 5
    uint64_t metadata_count = 5;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&metadata_count),
                  reinterpret_cast<uint8_t *>(&metadata_count) + 8);

    file.writeBytes(header);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);

    // Should fail with partial implementation message or parsing error (not crash)
    EXPECT_TRUE(result.isErr());
}

TEST_F(GGUFHeaderTest, InvalidMagic) {
    TempFile file(".gguf");

    std::vector<uint8_t> header;
    uint32_t             bad_magic = 0x12345678;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&bad_magic),
                  reinterpret_cast<uint8_t *>(&bad_magic) + 4);

    file.writeBytes(header);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);

    EXPECT_TRUE(result.isErr());
    // Error message should indicate header or magic issue
    EXPECT_TRUE(result.error().find("header") != std::string::npos ||
                result.error().find("magic") != std::string::npos ||
                result.error().find("Invalid") != std::string::npos);
}

TEST_F(GGUFHeaderTest, UnsupportedVersion) {
    TempFile file(".gguf");

    std::vector<uint8_t> header;
    uint32_t             magic = GGUF_MAGIC;
    uint32_t             bad_version = 99;

    header.insert(header.end(), reinterpret_cast<uint8_t *>(&magic),
                  reinterpret_cast<uint8_t *>(&magic) + 4);
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&bad_version),
                  reinterpret_cast<uint8_t *>(&bad_version) + 4);

    file.writeBytes(header);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);

    EXPECT_TRUE(result.isErr());
    // Error message should indicate version issue
    EXPECT_TRUE(result.error().find("version") != std::string::npos ||
                result.error().find("Unsupported") != std::string::npos ||
                result.error().find("header") != std::string::npos);
}

// Unit tests for binary format
class BinLoaderTest : public ::testing::Test {};

TEST_F(BinLoaderTest, InvalidMagic) {
    TempFile file(".bin");

    uint32_t bad_magic = 0x12345678;
    file.write(bad_magic);

    ModelConfig config;
    auto        result = ModelLoader::loadBin(file.path(), config);

    EXPECT_TRUE(result.isErr());
    EXPECT_TRUE(result.error().find("magic") != std::string::npos);
}

TEST_F(BinLoaderTest, FileNotFound) {
    ModelConfig config;
    auto        result = ModelLoader::loadBin("/nonexistent/path/model.bin", config);

    EXPECT_TRUE(result.isErr());
    EXPECT_TRUE(result.error().find("open") != std::string::npos ||
                result.error().find("Failed") != std::string::npos);
}

TEST_F(BinLoaderTest, RuntimeLoadRejectsGGUFPath) {
    ModelConfig config;
    auto        result = InferenceEngine::load("model.gguf", config);

    EXPECT_TRUE(result.isErr());
    EXPECT_NE(result.error().find("GGUF runtime loading is not supported yet"), std::string::npos);
}

TEST_F(BinLoaderTest, TruncatedHeader) {
    TempFile file(".bin");

    // Write only magic, no version
    uint32_t magic = BIN_MAGIC;
    file.write(magic);

    ModelConfig config;
    auto        result = ModelLoader::loadBin(file.path(), config);

    EXPECT_TRUE(result.isErr());
}

// Property-based tests for corrupted file handling
// Feature: tiny-llm-inference-engine, Property 9: Corrupted File Error Handling
// Validates: Requirements 1.5

#if 0
// NOTE: Property-based tests are temporarily disabled due to GCC 11/12
// compatibility issues with rapidcheck's GTest integration.

RC_GTEST_PROP(CorruptedFileProperty, RandomBytesReturnError, (std::vector<uint8_t> random_bytes)) {
    // Skip empty files
    RC_PRE(!random_bytes.empty());

    TempFile file(".bin");
    file.writeBytes(random_bytes);

    ModelConfig config;
    auto        result = ModelLoader::loadBin(file.path(), config);

    // Property: random bytes should always result in error, never crash
    // The result should be an error (unless by extreme chance it's valid)
    if (random_bytes.size() < sizeof(BinHeader)) {
        RC_ASSERT(result.isErr());
    }
    // Even if it passes magic check, it should fail somewhere
}

RC_GTEST_PROP(CorruptedFileProperty, TruncatedFilesReturnError, (int truncate_at)) {
    // Create a minimal valid-looking header
    std::vector<uint8_t> data;

    uint32_t    magic = BIN_MAGIC;
    uint32_t    version = BIN_VERSION;
    ModelConfig config;

    data.insert(data.end(), reinterpret_cast<uint8_t *>(&magic),
                reinterpret_cast<uint8_t *>(&magic) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&version),
                reinterpret_cast<uint8_t *>(&version) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&config),
                reinterpret_cast<uint8_t *>(&config) + sizeof(config));

    // Truncate at random position
    truncate_at = std::abs(truncate_at) % (data.size() + 1);
    data.resize(truncate_at);

    TempFile file(".bin");
    file.writeBytes(data);

    auto result = ModelLoader::loadBin(file.path(), config);

    // Property: truncated files should return error
    RC_ASSERT(result.isErr());
}

RC_GTEST_PROP(CorruptedFileProperty, WrongVersionReturnError, (uint32_t bad_version)) {
    // Ensure version is not valid
    RC_PRE(bad_version != BIN_VERSION);

    std::vector<uint8_t> data;

    uint32_t magic = BIN_MAGIC;
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&magic),
                reinterpret_cast<uint8_t *>(&magic) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&bad_version),
                reinterpret_cast<uint8_t *>(&bad_version) + 4);

    TempFile file(".bin");
    file.writeBytes(data);

    ModelConfig config;
    auto        result = ModelLoader::loadBin(file.path(), config);

    // Property: wrong version should return error
    RC_ASSERT(result.isErr());
    RC_ASSERT(result.error().find("version") != std::string::npos);
}

RC_GTEST_PROP(CorruptedFileProperty, GGUFRandomBytesReturnError,
              (std::vector<uint8_t> random_bytes)) {
    RC_PRE(!random_bytes.empty());

    TempFile file(".gguf");
    file.writeBytes(random_bytes);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);

    // Property: random bytes should result in error for GGUF too
    // Unless by chance they form valid GGUF magic
    if (random_bytes.size() < 4 ||
        *reinterpret_cast<const uint32_t *>(random_bytes.data()) != GGUF_MAGIC) {
        RC_ASSERT(result.isErr());
    }
}
#endif

// Test dimension mismatch
TEST_F(BinLoaderTest, DimensionMismatch) {
    TempFile file(".bin");

    std::vector<uint8_t> data;

    uint32_t magic = BIN_MAGIC;
    uint32_t version = BIN_VERSION;

    // Create config with different dimensions
    ModelConfig stored_config;
    stored_config.hidden_dim = 256;
    stored_config.num_layers = 2;
    stored_config.vocab_size = 1000;

    data.insert(data.end(), reinterpret_cast<uint8_t *>(&magic),
                reinterpret_cast<uint8_t *>(&magic) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&version),
                reinterpret_cast<uint8_t *>(&version) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&stored_config),
                reinterpret_cast<uint8_t *>(&stored_config) + sizeof(stored_config));

    file.writeBytes(data);

    // Try to load with different config
    ModelConfig expected_config;
    expected_config.hidden_dim = 512; // Different!
    expected_config.num_layers = 2;
    expected_config.vocab_size = 1000;

    auto result = ModelLoader::loadBin(file.path(), expected_config);

    EXPECT_TRUE(result.isErr());
    EXPECT_TRUE(result.error().find("mismatch") != std::string::npos);
}
