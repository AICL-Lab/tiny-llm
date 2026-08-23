#!/usr/bin/env python3
"""生成 include/tiny_llm/unicode_tables.h（tokenizer 预分词用的 Unicode 类别区间表）。

数据源：unicodedata + str.isspace（= Unicode White_Space 属性）。
重新生成：python3 scripts/gen_unicode_tables.py
"""
import unicodedata
from pathlib import Path

OUT = Path("include/tiny_llm/unicode_tables.h")
EXPECTED_UNICODE_VERSION = "15.0.0"


def compact_ranges(cps):
    """把升序码点列表压缩为 [first, last] 闭区间。"""
    ranges = []
    start = prev = None
    for cp in cps:
        if start is None:
            start = prev = cp
        elif cp == prev + 1:
            prev = cp
        else:
            ranges.append((start, prev))
            start = prev = cp
    if start is not None:
        ranges.append((start, prev))
    return ranges


def cat_set(prefix):
    return [cp for cp in range(0x110000) if unicodedata.category(chr(cp))[0] == prefix]


def fmt_ranges(ranges):
    lines = ["    {0x%X, 0x%X}," % r for r in ranges]
    return "\n".join(lines)


# Unicode White_Space 属性全集（25 个码点）。
# 注意：CPython str.isspace() 额外把 U+001C..U+001F 当作空白，但 Unicode
# White_Space 属性不含它们，Rust regex \s 也以 White_Space 为准，故显式列出。
WHITE_SPACE = [
    0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x0020, 0x0085, 0x00A0,
    0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
    0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F,
    0x3000,
]


def main():
    if unicodedata.unidata_version != EXPECTED_UNICODE_VERSION:
        raise RuntimeError(
            "Unicode database version mismatch: expected "
            f"{EXPECTED_UNICODE_VERSION}, got {unicodedata.unidata_version}"
        )

    letters = compact_ranges(cat_set("L"))
    numbers = compact_ranges(cat_set("N"))
    ws = WHITE_SPACE

    ws_cells = ["0x%04X" % cp for cp in ws]
    ws_lines = []
    for i in range(0, len(ws_cells), 8):
        ws_lines.append("    " + ", ".join(ws_cells[i:i + 8]) + ",")

    OUT.write_text(
        f"// 由脚本从 Unicode {EXPECTED_UNICODE_VERSION} 类别数据生成的紧凑区间表（tokenizer 预分词用）。\n"
        "// 重新生成方式见 scripts/gen_unicode_tables.py。\n"
        "#pragma once\n\n"
        "#include <cstdint>\n\n"
        "namespace tiny_llm {\n\n"
        "struct CodepointRange {\n"
        "    uint32_t first;\n"
        "    uint32_t last; // inclusive\n"
        "};\n\n"
        "// \\p{L} = Lu | Ll | Lt | Lm | Lo\n"
        "// clang-format off\n"
        "inline constexpr CodepointRange kLetterRanges[] = {\n"
        f"{fmt_ranges(letters)}\n"
        "};\n\n"
        "// \\p{N} = Nd | Nl | No\n"
        "inline constexpr CodepointRange kNumberRanges[] = {\n"
        f"{fmt_ranges(numbers)}\n"
        "};\n\n"
        "// Unicode White_Space 属性全集\n"
        "inline constexpr uint32_t kWhitespace[] = {\n"
        + "\n".join(ws_lines)
        + "\n};\n\n"
        "// clang-format on\n\n"
        "} // namespace tiny_llm\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}: L={len(letters)} ranges, N={len(numbers)} ranges, ws={len(ws)}")


if __name__ == "__main__":
    main()
