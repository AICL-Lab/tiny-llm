#include "tiny_llm/tokenizer.h"

#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/unicode_tables.h"

#include <algorithm>
#include <array>
#include <climits>

namespace tiny_llm {

namespace {

// GGML token_type 取值（llama.cpp 约定）
constexpr int kTokenNormal = 1;
constexpr int kTokenControl = 3;
constexpr int kTokenUserDefined = 4;

// ── UTF-8 解码/编码 ────────────────────────────────────────────

struct Cp {
    uint32_t cp;
    size_t   off; // 该码点在原文中的字节偏移
};

std::vector<Cp> decodeCps(const std::string &s) {
    std::vector<Cp> out;
    size_t          i = 0;
    while (i < s.size()) {
        uint8_t b = static_cast<uint8_t>(s[i]);
        size_t  len = 1;
        if (b >= 0xF0)
            len = 4;
        else if (b >= 0xE0)
            len = 3;
        else if (b >= 0xC0)
            len = 2;
        // continuation 校验：多字节序列的后续字节必须都是 0x80-0xBF，
        // 否则 leader 单独按单字节回退（与 HF GPT-2 byte-level 的逐字节
        // 语义一致）。此前 leader+非continuation 会被拼成伪码点（如
        // \xC2\x41 -> U+0141），导致预分词切分点与 HF 不一致。
        auto isContinuation = [&](size_t k) {
            return k < s.size() && (static_cast<uint8_t>(s[k]) & 0xC0) == 0x80;
        };
        for (size_t k = 1; k < len; ++k) {
            if (!isContinuation(i + k)) {
                len = 1;
                break;
            }
        }
        if (i + len > s.size()) len = 1; // 残缺尾部：leader 单独成字节，剩余字节由后续迭代处理
        // 修复：len==1（ASCII 或孤立 continuation byte 0x80-0xBF）时用
        // 无符号字节值。此前 `uint32_t cp = s[i]` 在 char 为有符号时会把
        // 0x80-0xBF 符号扩展成 0xFFFFFF80，导致后续 isLetter/预分词查表
        // 错分类。len>=2 分支随后覆盖 cp，不受影响。
        uint32_t cp = b;
        if (len == 2)
            cp = ((b & 0x1Fu) << 6) | (static_cast<uint8_t>(s[i + 1]) & 0x3Fu);
        else if (len == 3)
            cp = ((b & 0x0Fu) << 12) | ((static_cast<uint8_t>(s[i + 1]) & 0x3Fu) << 6) |
                 (static_cast<uint8_t>(s[i + 2]) & 0x3Fu);
        else if (len == 4)
            cp = ((b & 0x07u) << 18) | ((static_cast<uint8_t>(s[i + 1]) & 0x3Fu) << 12) |
                 ((static_cast<uint8_t>(s[i + 2]) & 0x3Fu) << 6) |
                 (static_cast<uint8_t>(s[i + 3]) & 0x3Fu);
        out.push_back({cp, i});
        i += len;
    }
    return out;
}

void appendUtf8(std::string &s, uint32_t cp) {
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
}

// ── Unicode 类别判定（查 unicode_tables.h 区间表） ─────────────

bool inRanges(uint32_t cp, const CodepointRange *ranges, size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp > ranges[mid].last)
            lo = mid + 1;
        else if (cp < ranges[mid].first)
            hi = mid;
        else
            return true;
    }
    return false;
}

bool isLetter(uint32_t cp) { return inRanges(cp, kLetterRanges, std::size(kLetterRanges)); }
bool isNumber(uint32_t cp) { return inRanges(cp, kNumberRanges, std::size(kNumberRanges)); }

bool isWhitespace(uint32_t cp) {
    const auto *p = std::lower_bound(kWhitespace, std::end(kWhitespace), cp);
    return p != std::end(kWhitespace) && *p == cp;
}

// ── GPT-2 字节编码表 ──────────────────────────────────────────

// 与 GPT-2 bytes_to_unicode() 一致：33-126 / 161-172 / 174-255 映射到自身，
// 其余 68 个字节按升序映射到 256, 257, ...
std::array<uint32_t, 256> makeByteToUnicode() {
    std::array<uint32_t, 256> table{};
    std::array<bool, 256>     printable{};
    for (uint32_t b = 33; b <= 126; ++b)
        printable[b] = true;
    for (uint32_t b = 161; b <= 172; ++b)
        printable[b] = true;
    for (uint32_t b = 174; b <= 255; ++b)
        printable[b] = true;
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; ++b)
        table[b] = printable[b] ? b : 256 + n++;
    return table;
}

const std::array<uint32_t, 256> &byteTable() {
    static const std::array<uint32_t, 256> table = makeByteToUnicode();
    return table;
}

// 逆映射表：码点 -> 字节。码点最大 256+67=323，开 512 足够。
const std::array<int, 512> &inverseByteTable() {
    static const std::array<int, 512> inv = [] {
        std::array<int, 512> a{};
        a.fill(-1);
        const auto &fwd = byteTable();
        for (uint32_t b = 0; b < 256; ++b)
            a[fwd[b]] = static_cast<int>(b);
        return a;
    }();
    return inv;
}

// 把 UTF-8 文本逐字节做 GPT-2 字节编码
std::string byteEncode(const std::string &utf8) {
    std::string out;
    out.reserve(utf8.size() * 2);
    const auto &table = byteTable();
    for (unsigned char c : utf8)
        appendUtf8(out, table[c]);
    return out;
}

bool ciEqualsLower(uint32_t cp, char lower) {
    return cp == static_cast<uint32_t>(lower) || cp == static_cast<uint32_t>(lower - 'a' + 'A');
}

// 缩写后缀表，顺序即正则分支 (?i:'s|'t|'re|'ve|'m|'ll|'d) 的优先级
constexpr std::string_view kContractions[] = {"s", "t", "re", "ve", "m", "ll", "d"};

} // namespace

const uint32_t *byteToUnicodeTable() { return byteTable().data(); }

int unicodeToByte(uint32_t cp) {
    if (cp >= inverseByteTable().size()) return -1;
    return inverseByteTable()[cp];
}

std::vector<uint32_t> decodeUtf8Codepoints(const std::string &s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (const Cp &c : decodeCps(s))
        out.push_back(c.cp);
    return out;
}

// ── 构建 ──────────────────────────────────────────────────────

Result<Tokenizer> Tokenizer::build(const TokenizerData &data) {
    if (data.model_type != "gpt2")
        return Result<Tokenizer>::err("unsupported tokenizer model: " + data.model_type);
    if (data.tokens.empty()) return Result<Tokenizer>::err("empty token list");

    Tokenizer t;
    t.id_to_token_ = data.tokens;
    t.bos_id_ = data.bos_token_id;
    t.eos_id_ = data.eos_token_id;
    for (size_t i = 0; i < data.tokens.size(); ++i) {
        if (!t.token_to_id_.emplace(data.tokens[i], static_cast<int>(i)).second)
            return Result<Tokenizer>::err("duplicate token: " + data.tokens[i]);
    }

    // BPE 起始符号是 256 个字节 token，缺失则词表不完整
    const auto &table = byteTable();
    for (uint32_t b = 0; b < 256; ++b) {
        std::string piece;
        appendUtf8(piece, table[b]);
        if (t.token_to_id_.find(piece) == t.token_to_id_.end())
            return Result<Tokenizer>::err("byte token missing from vocab: byte " +
                                          std::to_string(b));
    }

    for (size_t rank = 0; rank < data.merges.size(); ++rank) {
        const std::string &m = data.merges[rank];
        if (m.find(' ') == std::string::npos)
            return Result<Tokenizer>::err("malformed merge rule: " + m);
        if (!t.merge_rank_.emplace(m, static_cast<int>(rank)).second)
            return Result<Tokenizer>::err("duplicate merge rule: " + m);
    }

    // CONTROL / USER_DEFINED token 作为特殊 token 精确匹配隔离（HF added tokens 语义）
    for (size_t i = 0; i < data.tokens.size(); ++i) {
        int type = i < data.token_types.size() ? data.token_types[i] : kTokenNormal;
        if (type == kTokenControl || type == kTokenUserDefined)
            t.special_tokens_.push_back({data.tokens[i], static_cast<int>(i)});
    }
    std::stable_sort(t.special_tokens_.begin(), t.special_tokens_.end(),
                     [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });

    return Result<Tokenizer>::ok(std::move(t));
}

// ── 预分词：手写 Qwen2 正则（等价于 HF tokenizer.json 的 Split pattern） ──
//
// (?i:'s|'t|'re|'ve|'m|'ll|'d)
// | [^\r\n\p{L}\p{N}]?\p{L}+
// | \p{N}
// | ?[^\s\p{L}\p{N}]+[\r\n]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+

std::vector<std::string> Tokenizer::preTokenize(const std::string &text) const {
    std::vector<Cp> cps = decodeCps(text);
    const size_t    n = cps.size();

    auto endByte = [&](size_t idx) { return idx < n ? cps[idx].off : text.size(); };

    std::vector<std::string> out;
    size_t                   i = 0;
    while (i < n) {
        size_t   j = i; // 匹配终点（码点下标，不含）
        uint32_t c = cps[i].cp;

        // 分支 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'') {
            for (auto suf : kContractions) {
                size_t k = i + 1;
                bool   ok = suf.size() <= n - k;
                for (size_t s = 0; ok && s < suf.size(); ++s, ++k) {
                    if (!ciEqualsLower(cps[k].cp, suf[s])) ok = false;
                }
                if (ok) {
                    j = k;
                    break;
                }
            }
        }

        // 分支 2: [^\r\n\p{L}\p{N}]?\p{L}+
        if (j == i) {
            if (isLetter(c)) {
                j = i;
                while (j < n && isLetter(cps[j].cp))
                    ++j;
            } else if (c != '\r' && c != '\n' && !isLetter(c) && !isNumber(c) && i + 1 < n &&
                       isLetter(cps[i + 1].cp)) {
                j = i + 1;
                while (j < n && isLetter(cps[j].cp))
                    ++j;
            }
        }

        // 分支 3: \p{N}（单个数字码点）
        if (j == i && isNumber(c)) j = i + 1;

        // 分支 4: ' '?[^\s\p{L}\p{N}]+[\r\n]*
        if (j == i) {
            auto isOther = [](uint32_t cp) {
                return !isWhitespace(cp) && !isLetter(cp) && !isNumber(cp);
            };
            size_t k = i;
            if (k < n && cps[k].cp == ' ') ++k;
            if (k < n && isOther(cps[k].cp)) {
                while (k < n && isOther(cps[k].cp))
                    ++k;
                while (k < n && (cps[k].cp == '\r' || cps[k].cp == '\n'))
                    ++k;
                j = k;
            }
        }

        // 分支 5: \s*[\r\n]+（吃到空白段内最后一个换行为止）
        if (j == i) {
            size_t r = i;
            while (r < n && isWhitespace(cps[r].cp))
                ++r;
            size_t lastNl = n;
            for (size_t k = i; k < r; ++k) {
                if (cps[k].cp == '\r' || cps[k].cp == '\n') lastNl = k;
            }
            if (lastNl != n) j = lastNl + 1;
        }

        // 分支 6: \s+(?!\S)
        if (j == i) {
            size_t r = i;
            while (r < n && isWhitespace(cps[r].cp))
                ++r;
            if (r > i) {
                if (r == n)
                    j = r; // 文末空白整体成块
                else if (r - i >= 2)
                    j = r - 1; // 丢弃最后一个空白字符，留给下一轮
            }
        }

        // 分支 7: \s+
        if (j == i) {
            j = i;
            while (j < n && isWhitespace(cps[j].cp))
                ++j;
        }

        if (j == i) // 理论上不可达：任何码点必属 L/N/空白/其他 之一
            j = i + 1;

        out.push_back(text.substr(cps[i].off, endByte(j) - cps[i].off));
        i = j;
    }
    return out;
}

// ── BPE ───────────────────────────────────────────────────────

std::vector<std::string> Tokenizer::bpe(const std::string &word) const {
    // word 已是字节编码串；符号序列 = 其码点序列
    std::vector<std::string> syms;
    for (const Cp &c : decodeCps(word)) {
        syms.emplace_back();
        appendUtf8(syms.back(), c.cp);
    }

    while (syms.size() >= 2) {
        int    bestRank = INT_MAX;
        size_t bestPos = 0;
        for (size_t k = 0; k + 1 < syms.size(); ++k) {
            auto it = merge_rank_.find(syms[k] + " " + syms[k + 1]);
            if (it != merge_rank_.end() && it->second < bestRank) {
                bestRank = it->second;
                bestPos = k;
            }
        }
        if (bestRank == INT_MAX) break;
        const std::string        a = syms[bestPos], b = syms[bestPos + 1];
        std::vector<std::string> next;
        next.reserve(syms.size());
        for (size_t k = 0; k < syms.size();) {
            if (k + 1 < syms.size() && syms[k] == a && syms[k + 1] == b) {
                next.push_back(a + b);
                k += 2;
            } else {
                next.push_back(std::move(syms[k]));
                ++k;
            }
        }
        syms = std::move(next);
    }
    return syms;
}

// ── encode / decode ───────────────────────────────────────────

bool Tokenizer::pieceIsSpecial(int id) const {
    return std::any_of(special_tokens_.begin(), special_tokens_.end(),
                       [id](const auto &p) { return p.second == id; });
}

std::vector<int> Tokenizer::encode(const std::string &text, bool add_bos) const {
    std::vector<int> ids;
    if (add_bos && bos_id_ >= 0) ids.push_back(bos_id_);

    // 1) 特殊 token 精确匹配隔离（从左到右，同位置长者优先）
    size_t      i = 0;
    std::string seg;
    auto        flushSeg = [&] {
        if (seg.empty()) return;
        // 2) 预分词 -> 3) 字节编码 -> 4) BPE
        for (const std::string &chunk : preTokenize(seg)) {
            for (const std::string &piece : bpe(byteEncode(chunk))) {
                auto it = token_to_id_.find(piece);
                if (it != token_to_id_.end()) ids.push_back(it->second);
            }
        }
        seg.clear();
    };

    while (i < text.size()) {
        bool matched = false;
        for (const auto &[piece, id] : special_tokens_) {
            if (text.compare(i, piece.size(), piece) == 0) {
                flushSeg();
                ids.push_back(id);
                i += piece.size();
                matched = true;
                break;
            }
        }
        if (!matched) seg += text[i++];
    }
    flushSeg();
    return ids;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
        const std::string &piece = id_to_token_[id];
        if (pieceIsSpecial(id)) {
            out += piece;
            continue;
        }
        for (const Cp &c : decodeCps(piece)) {
            int b = unicodeToByte(c.cp);
            if (b >= 0)
                out += static_cast<char>(b);
            else
                appendUtf8(out, c.cp); // 非字节编码字符按原样输出
        }
    }
    return out;
}

// ── GGUF 元数据加载 ───────────────────────────────────────────

Result<TokenizerData> loadTokenizerData(const GGUFMetadata &md) {
    TokenizerData d;
    d.model_type = md.getOr<std::string>("tokenizer.ggml.model", std::string{});
    if (auto r = md.get<std::vector<std::string>>("tokenizer.ggml.tokens"); r.isOk())
        d.tokens = std::move(r.value());
    if (auto r = md.get<std::vector<std::string>>("tokenizer.ggml.merges"); r.isOk())
        d.merges = std::move(r.value());
    if (auto r = md.get<std::vector<int32_t>>("tokenizer.ggml.token_type"); r.isOk())
        d.token_types.assign(r.value().begin(), r.value().end());
    // R2: 特殊 token id 缺失时必须保持 -1（无效值），不能用 0 兜底——
    // 0 是合法 token id，add_bos=true 会向序列注入错误的 token。
    if (auto r = md.get<uint32_t>("tokenizer.ggml.bos_token_id"); r.isOk())
        d.bos_token_id = static_cast<int>(r.value());
    if (auto r = md.get<uint32_t>("tokenizer.ggml.eos_token_id"); r.isOk())
        d.eos_token_id = static_cast<int>(r.value());
    if (auto r = md.get<uint32_t>("tokenizer.ggml.padding_token_id"); r.isOk())
        d.padding_token_id = static_cast<int>(r.value());
    d.add_bos_token = md.getOr<bool>("tokenizer.ggml.add_bos_token", false);
    if (d.tokens.empty())
        return Result<TokenizerData>::err("tokenizer.ggml.tokens not found in GGUF metadata");
    return Result<TokenizerData>::ok(std::move(d));
}

} // namespace tiny_llm
