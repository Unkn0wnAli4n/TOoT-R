// oot-extract <game.nup> <out-dir>: pulls the newest ExeFS (main, sdk, subsdk0) out of a Switch 2
// .nup package and writes each NSO as a flat memory image (<module>.img: text/ro/data at their load
// offsets, bss zeroed) plus <module>.img.text-size (end of the executable segment, hex).
// NUP members are standard zstd; NSO segments are LZ4 or ZBIC (zstd with BIC-coded entropy tables).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <lz4.h>
#include <zstd.h>

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

int DecompressZbic(std::span<u8> dst, std::span<const u8> src); // zbic.cpp

namespace {

[[noreturn]] void Fail(const std::string& message) {
    std::fprintf(stderr, "oot-extract: %s\n", message.c_str());
    std::exit(1);
}

template <typename T> T Get(const std::vector<u8>& d, size_t at) {
    if (at + sizeof(T) > d.size()) Fail("truncated table of contents");
    T v;
    std::memcpy(&v, d.data() + at, sizeof(T));
    return v;
}

std::vector<u8> ReadAt(std::ifstream& f, u64 offset, u64 size) {
    std::vector<u8> out(size);
    f.seekg((std::streamoff)offset);
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)size);
    if (!f) Fail("read past the end of the package (incomplete file?)");
    return out;
}

void WriteFile(const std::filesystem::path& path, const void* data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(data), (std::streamsize)size);
    if (!out) Fail("cannot write " + path.string());
}

// NSO -> flat image (see the NSO0 header layout: segments at 0x10, compressed sizes at 0x60).
void NsoToImage(const std::vector<u8>& d, const std::filesystem::path& out) {
    if (d.size() < 0x100 || std::memcmp(d.data(), "NSO0", 4)) Fail(out.filename().string() + ": not an NSO");
    const u32 flags = Get<u32>(d, 0xc);
    u32 seg[3][3], csz[3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) seg[i][j] = Get<u32>(d, 0x10 + i * 0x10 + j * 4);
    for (int i = 0; i < 3; ++i) csz[i] = Get<u32>(d, 0x60 + i * 4);
    const u32 bss = Get<u32>(d, 0x3c);
    std::vector<u8> img((size_t)seg[2][1] + seg[2][2] + bss);
    for (int i = 0; i < 3; ++i) {
        const size_t source_size = (flags >> i & 1) ? csz[i] : seg[i][2];
        if ((size_t)seg[i][0] + source_size > d.size() || (size_t)seg[i][1] + seg[i][2] > img.size())
            Fail("segment outside the NSO");
        if (!seg[i][2]) continue;
        std::span<const u8> src{d.data() + seg[i][0], source_size};
        std::span<u8> dst{img.data() + seg[i][1], seg[i][2]};
        if (!(flags >> i & 1)) { std::memcpy(dst.data(), src.data(), src.size()); continue; }
        const int r = (flags >> 7 & 1) ? DecompressZbic(dst, src)
            : LZ4_decompress_safe(reinterpret_cast<const char*>(src.data()), reinterpret_cast<char*>(dst.data()),
                                  (int)src.size(), (int)dst.size());
        if (r != (int)seg[i][2]) Fail("cannot decompress segment of " + out.filename().string());
    }
    WriteFile(out, img.data(), img.size());
    char text_end[32];
    const int n = std::snprintf(text_end, sizeof text_end, "%x\n", seg[0][1] + seg[0][2]);
    WriteFile(out.string() + ".text-size", text_end, (size_t)n);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: oot-extract <game.nup> <out-dir>\n");
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) Fail(std::string("cannot open ") + argv[1]);
    const std::vector<u8> super = ReadAt(f, 0, 0x80);
    if (std::memcmp(super.data(), "NUP\x01", 4)) Fail("not a .nup package (expected the NUP header)");
    const u64 toc_off = Get<u64>(super, 8), toc_size = Get<u64>(super, 16);
    if (toc_size > (64u << 20)) Fail("implausible table of contents");
    const std::vector<u8> toc = ReadAt(f, toc_off, toc_size);
    const u32 packages = Get<u32>(toc, 4);
    const u64 dirs = Get<u64>(toc, 8), files = Get<u64>(toc, 16), pool = Get<u64>(toc, 24);
    auto name = [&](u32 off, u32 len) {
        if ((u64)pool + off + len > toc.size()) Fail("bad name");
        return std::string(reinterpret_cast<const char*>(toc.data() + pool + off), len);
    };
    size_t p = 0x30;
    struct Package { u64 title; u32 exefs; u32 version; u8 kind; };
    std::vector<Package> pkgs;
    for (u32 i = 0; i < packages; ++i, p += 0x40)
        pkgs.push_back({Get<u64>(toc, p), Get<u32>(toc, p + 0xc), Get<u32>(toc, p + 0x14), toc.at(p + 0x18)});
    const size_t dir_table = p, file_table = dir_table + dirs * 0x18, children = file_table + files * 0x50;

    // The newest program package with an ExeFS (the update when the dump bundles one).
    const Package* best = nullptr;
    for (const auto& k : pkgs)
        if (k.exefs != 0xffffffff && (k.kind == 0 || k.kind == 1) && (!best || k.version > best->version)) best = &k;
    if (!best) Fail("no program ExeFS in this package");
    std::printf("package %016llX v%u\n", (unsigned long long)best->title, best->version);

    const size_t d = dir_table + (size_t)best->exefs * 0x18;
    const u32 first = Get<u32>(toc, d + 8), count = Get<u32>(toc, d + 12);
    const std::filesystem::path out_dir = argv[2];
    std::filesystem::create_directories(out_dir);
    int found = 0;
    for (u32 i = 0; i < count; ++i) {
        const u32 child = Get<u32>(toc, children + 4 * (size_t)(first + i));
        if (child < dirs) continue;
        const size_t e = file_table + (size_t)(child - dirs) * 0x50;
        const std::string fname = name(Get<u32>(toc, e), Get<u32>(toc, e + 4));
        if (fname != "main" && fname != "sdk" && fname != "subsdk0") continue;
        const u64 size = Get<u64>(toc, e + 8), off = Get<u64>(toc, e + 0x10), stored = Get<u64>(toc, e + 0x18);
        const u8 codec = toc.at(e + 0x40);
        std::vector<u8> data = ReadAt(f, off, stored);
        if (codec == 1) {
            std::vector<u8> raw(size);
            const size_t r = ZSTD_decompress(raw.data(), raw.size(), data.data(), data.size());
            if (ZSTD_isError(r) || r != size) Fail("cannot decompress " + fname);
            data.swap(raw);
        } else if (codec != 0) {
            Fail("unknown codec for " + fname);
        }
        NsoToImage(data, out_dir / (fname + ".img"));
        std::printf("  %s.img\n", fname.c_str());
        ++found;
    }
    if (found != 3) Fail("the ExeFS lacks main/sdk/subsdk0");
    return 0;
}
