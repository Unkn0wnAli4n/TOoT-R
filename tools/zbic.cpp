// ZBIC decompression: kinnay/zbic's single-file zstd (third_party/zbic) built with ZSTD_ZBIC_SUPPORT,
// every symbol static so it can coexist with the system libzstd in one program.
#include <cstdint>
#include <span>

#define ZSTD_ZBIC_SUPPORT 1
#define ZSTDLIB_VISIBLE static
#define ZSTDLIB_HIDDEN static
#define ZSTDERRORLIB_VISIBLE static
#define ZSTDERRORLIB_HIDDEN static
#undef ZSTD_MULTITHREAD
#define g_ZSTD_threading_useless_symbol g_ZSTD_zbic_threading_useless_symbol
#include "zstd.c"
#undef g_ZSTD_threading_useless_symbol

int DecompressZbic(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) {
    if (dst.empty() || src.empty()) return -1;
    const size_t r = ZSTD_decompress(dst.data(), dst.size(), src.data(), src.size());
    return ZSTD_isError(r) ? -1 : static_cast<int>(r);
}
