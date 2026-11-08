// nn::fs natively: "rom" mounts read the game's RomFS straight out of the .nup (base RomFS with the
// update's files overlaid), save data and temporary storage live in host directories.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <unordered_map>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../native.h"

namespace fs = std::filesystem;
using namespace rt;

namespace {

constexpr u32 ResultPathNotFound = 0x202;
constexpr u32 ResultPathAlreadyExists = 0x402;
constexpr u32 ResultMountNameAlreadyExists = 0xBC02;
constexpr u32 ResultNotMounted = 0xDE02; // fs::ResultTargetNotFound-like
constexpr u32 ResultUnsupported = 0x31C02;

// ---- RomFS from the .nup ----------------------------------------------------------------------
struct RomEntry {
    bool directory;
    u64 offset = 0, size = 0;
    std::map<std::string, std::string> children; // name -> full path (directories)
};

struct NupRomFs {
    int fd = -1;
    std::unordered_map<std::string, RomEntry> entries; // "/a/b" -> entry; "/" is the root

    static std::string Join(const std::string& dir, const std::string& name) {
        return dir == "/" ? "/" + name : dir + "/" + name;
    }

    void Open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) Fatal("cannot open game package %s", path.c_str());
        u8 super[0x80];
        if (pread(fd, super, sizeof super, 0) != (ssize_t)sizeof super || std::memcmp(super, "NUP\x01", 4))
            Fatal("%s is not a NUP package", path.c_str());
        u64 toc_off, toc_size;
        std::memcpy(&toc_off, super + 8, 8);
        std::memcpy(&toc_size, super + 16, 8);
        std::vector<u8> toc(toc_size);
        if (pread(fd, toc.data(), toc_size, toc_off) != (ssize_t)toc_size) Fatal("short TOC read");
        u32 packages;
        u64 dirs, files, pool_off;
        std::memcpy(&packages, &toc[4], 4);
        std::memcpy(&dirs, &toc[8], 8);
        std::memcpy(&files, &toc[16], 8);
        std::memcpy(&pool_off, &toc[24], 8);
        const u8* p = toc.data() + 0x30;
        struct Pkg { u64 title; u32 romfs; u32 version; u8 kind; };
        std::vector<Pkg> pkgs;
        for (u32 i = 0; i < packages; ++i, p += 0x40) {
            Pkg k;
            std::memcpy(&k.title, p, 8);
            std::memcpy(&k.romfs, p + 0x10, 4);
            std::memcpy(&k.version, p + 0x14, 4);
            k.kind = p[0x18];
            pkgs.push_back(k);
        }
        struct Dir { std::string name; u32 first, count; };
        std::vector<Dir> dir_list(dirs);
        for (u64 i = 0; i < dirs; ++i, p += 0x18) {
            u32 no, nl;
            std::memcpy(&no, p, 4);
            std::memcpy(&nl, p + 4, 4);
            std::memcpy(&dir_list[i].first, p + 8, 4);
            std::memcpy(&dir_list[i].count, p + 12, 4);
            dir_list[i].name.assign(reinterpret_cast<const char*>(&toc[pool_off + no]), nl);
        }
        struct File { std::string name; u64 size, offset; u8 codec; };
        std::vector<File> file_list(files);
        for (u64 i = 0; i < files; ++i, p += 0x50) {
            u32 no, nl;
            std::memcpy(&no, p, 4);
            std::memcpy(&nl, p + 4, 4);
            std::memcpy(&file_list[i].size, p + 8, 8);
            std::memcpy(&file_list[i].offset, p + 16, 8);
            file_list[i].codec = p[0x40];
            file_list[i].name.assign(reinterpret_cast<const char*>(&toc[pool_off + no]), nl);
        }
        const u32* children = reinterpret_cast<const u32*>(p);
        entries["/"] = RomEntry{true};
        // Base first, then updates (higher versions win).
        std::sort(pkgs.begin(), pkgs.end(), [](auto& a, auto& b) { return a.kind != b.kind ? a.kind < b.kind : a.version < b.version; });
        size_t compressed = 0;
        const auto add_dir = [&](auto&& self, u32 dir, const std::string& path) -> void {
            for (u32 i = 0; i < dir_list[dir].count; ++i) {
                const u32 child = children[dir_list[dir].first + i];
                if (child < dirs) {
                    const std::string sub = Join(path, dir_list[child].name);
                    entries[path].children[dir_list[child].name] = sub;
                    if (!entries.count(sub)) entries[sub] = RomEntry{true};
                    self(self, child, sub);
                } else {
                    const File& f = file_list[child - dirs];
                    if (f.codec != 0) ++compressed;
                    const std::string sub = Join(path, f.name);
                    entries[path].children[f.name] = sub;
                    entries[sub] = RomEntry{false, f.offset, f.size};
                }
            }
        };
        for (const auto& k : pkgs)
            if (k.romfs != 0xffffffffu && (k.kind == 0 || k.kind == 1)) add_dir(add_dir, k.romfs, "/");
        if (compressed) Fatal("%zu compressed RomFS files are not supported yet", compressed);
        RT_LOG("RomFS: %zu entries from %s", entries.size(), path.c_str());
    }

    const RomEntry* Find(const std::string& path) const {
        auto it = entries.find(path.empty() ? "/" : path);
        return it == entries.end() ? nullptr : &it->second;
    }
};

NupRomFs& Rom() {
    static NupRomFs rom;
    static std::once_flag once;
    std::call_once(once, [] {
        std::string path;
        if (const char* env = std::getenv("OOT_RECOMP_NUP")) path = env;
        else path = DataDir() + "/game.nup";
        rom.Open(path);
    });
    return rom;
}

// ---- mounts and handles ------------------------------------------------------------------------
struct Mount {
    bool rom;
    fs::path host; // for host mounts
};

std::mutex fs_mutex;
std::unordered_map<std::string, Mount> mounts;

fs::path DataRoot() {
    if (const char* env = std::getenv("OOT_RECOMP_SAVE")) return env;
    return fs::path(DataDir());
}

// Splits "name:/path" into the mount and an absolute path inside it.
bool Split(const char* full, Mount& mount, std::string& path) {
    const std::string s = full;
    const auto colon = s.find(':');
    if (colon == std::string::npos) return false;
    std::lock_guard lock{fs_mutex};
    auto it = mounts.find(s.substr(0, colon));
    if (it == mounts.end()) return false;
    mount = it->second;
    path = s.substr(colon + 1);
    if (path.empty() || path[0] != '/') path = "/" + path;
    // SDK paths may join a mount root ending in '/' with an absolute resource
    // path. RomFS keys use one separator, just like the console filesystem.
    path.erase(std::unique(path.begin(), path.end(), [](char a, char b) {
        return a == '/' && b == '/';
    }), path.end());
    path = fs::path(path).lexically_normal().generic_string();
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    return true;
}

struct OpenFile {
    bool rom;
    u64 rom_offset = 0, rom_size = 0;
    int host_fd = -1;
    int mode = 0;
};
struct OpenDir {
    std::vector<std::pair<std::string, std::pair<bool, u64>>> items; // name -> (is_dir, size)
    size_t next = 0;
};

std::atomic<u64> next_handle{1};
std::unordered_map<u64, OpenFile> files;
std::unordered_map<u64, OpenDir> directories;

u32 AddMount(const char* name, Mount m) {
    std::lock_guard lock{fs_mutex};
    if (mounts.count(name)) return ResultMountNameAlreadyExists;
    mounts.emplace(name, std::move(m));
    return 0;
}

u32 HostMount(const char* name, const fs::path& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    return AddMount(name, Mount{false, dir});
}

fs::path HostPath(const Mount& m, const std::string& path) { return m.host / path.substr(1); }

} // namespace

// ---- mounting ----------------------------------------------------------------------------------
NATIVE("_ZN2nn2fs22QueryMountRomCacheSizeEPm") { *Ptr<u64>(c, 0) = 0x1000; Ret(c, 0); }
NATIVE("_ZN2nn2fs8MountRomEPKcPvm") { Rom(); Ret(c, AddMount(Ptr<const char>(c, 0), Mount{true, {}})); }
NATIVE("_ZN2nn2fs8MountRomEPKc") { Rom(); Ret(c, AddMount(Ptr<const char>(c, 0), Mount{true, {}})); }
NATIVE("_ZN2nn2fs13MountSaveDataEPKcRKNS_7account3UidE") { Ret(c, HostMount(Ptr<const char>(c, 0), DataRoot() / "save")); }
NATIVE("_ZN2nn2fs21MountTemporaryStorageEPKc") { Ret(c, HostMount(Ptr<const char>(c, 0), DataRoot() / "temporary")); }
NATIVE("_ZN2nn2fs7UnmountEPKc") {
    std::lock_guard lock{fs_mutex};
    mounts.erase(Ptr<const char>(c, 0));
}
NATIVE("_ZN2nn2fs6CommitEPKc") {
#ifndef _WIN32
    sync();
#endif
    Ret(c, 0);
}
NATIVE("_ZN2nn2fs12SetAllocatorEPFPvmEPFvS1_mE") {}
NATIVE("_ZN2nn2fs26SetPriorityOnCurrentThreadENS0_8PriorityE") {}
NATIVE("_ZN2nn2fs25GetCurrentThreadFsContextEv") {
    static u64 context[8] = {};
    Ret(c, reinterpret_cast<u64>(context));
}
NATIVE("_ZN2nn2fs25SetCurrentThreadFsContextEPKNS0_9FsContextE") {}

// ---- files ---------------------------------------------------------------------------------------
NATIVE("_ZN2nn2fs8OpenFileEPNS0_10FileHandleEPKci") {
    static const bool trace = std::getenv("OOT_FS_TRACE") != nullptr;
    if (trace) RT_LOG("fs open %s", Ptr<const char>(c, 1));
    Mount m;
    std::string path;
    if (!Split(Ptr<const char>(c, 1), m, path)) return Ret(c, ResultPathNotFound);
    const int mode = (int)c->x[2];
    OpenFile f{m.rom};
    f.mode = mode;
    if (m.rom) {
        const RomEntry* e = Rom().Find(path);
        if (!e || e->directory) {
            if (trace) RT_LOG("fs missing %s", path.c_str());
            return Ret(c, ResultPathNotFound);
        }
        f.rom_offset = e->offset;
        f.rom_size = e->size;
    } else {
        const int flags = (mode & 2) ? O_RDWR : O_RDONLY;
        f.host_fd = ::open(HostPath(m, path).string().c_str(), flags | O_CLOEXEC);
        if (f.host_fd < 0) return Ret(c, ResultPathNotFound);
    }
    const u64 h = next_handle++;
    {
        std::lock_guard lock{fs_mutex};
        files.emplace(h, f);
    }
    *Ptr<u64>(c, 0) = h;
    Ret(c, 0);
}

static OpenFile* FindFile(u64 h) {
    std::lock_guard lock{fs_mutex};
    auto it = files.find(h);
    return it == files.end() ? nullptr : &it->second;
}

namespace nvn { void PrepareHostWrite(u64 address, u64 size); }
// pread into guest memory. Pages of GPU pools are write-protected to find CPU writes; the kernel
// does not fault on them but fails with EFAULT, so make them writable first (and again if
// a GPU submission protected them meanwhile).
static ssize_t GuestPread(int fd, void* buffer, u64 size, s64 offset) {
    u64 done = 0;
    for (int attempts = 0; done < size;) {
        nvn::PrepareHostWrite(reinterpret_cast<u64>(buffer) + done, size - done);
        const ssize_t n = pread(fd, static_cast<u8*>(buffer) + done, size - done, offset + (s64)done);
        if (n < 0) { if (errno == EFAULT && ++attempts < 64) continue; return done ? (ssize_t)done : n; }
        if (n == 0) break;
        done += (u64)n;
    }
    return (ssize_t)done;
}
static u32 DoRead(u64 h, s64 offset, void* buffer, u64 size, u64* read) {
    OpenFile* f = FindFile(h);
    if (!f) Fatal("nn::fs::ReadFile on an unknown handle");
    ssize_t n;
    if (f->rom) {
        if ((u64)offset > f->rom_size) return 0x7802; // OutOfRange-like
        size = std::min<u64>(size, f->rom_size - offset);
        n = GuestPread(Rom().fd, buffer, size, f->rom_offset + offset);
    } else {
        n = GuestPread(f->host_fd, buffer, size, offset);
    }
    if (n < 0) return 0x7802;
    if (read) *read = (u64)n;
    return 0;
}

NATIVE("_ZN2nn2fs8ReadFileENS0_10FileHandleElPvm") {
    Ret(c, DoRead(c->x[0], (s64)c->x[1], Ptr(c, 2), c->x[3], nullptr));
}
NATIVE("_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvm") {
    Ret(c, DoRead(c->x[1], (s64)c->x[2], Ptr(c, 3), c->x[4], Ptr<u64>(c, 0)));
}
NATIVE("_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvmRKNS0_10ReadOptionE") {
    Ret(c, DoRead(c->x[1], (s64)c->x[2], Ptr(c, 3), c->x[4], Ptr<u64>(c, 0)));
}
NATIVE("_ZN2nn2fs9WriteFileENS0_10FileHandleElPKvmRKNS0_11WriteOptionE") {
    OpenFile* f = FindFile(c->x[0]);
    if (!f || f->rom) Fatal("nn::fs::WriteFile on a read-only file");
    const ssize_t n = pwrite(f->host_fd, Ptr(c, 2), c->x[3], (off_t)c->x[1]);
    Ret(c, n == (ssize_t)c->x[3] ? 0 : 0x7802);
}
NATIVE("_ZN2nn2fs9FlushFileENS0_10FileHandleE") {
    if (OpenFile* f = FindFile(c->x[0]); f && !f->rom) fsync(f->host_fd);
    Ret(c, 0);
}
NATIVE("_ZN2nn2fs9CloseFileENS0_10FileHandleE") {
    std::lock_guard lock{fs_mutex};
    auto it = files.find(c->x[0]);
    if (it == files.end()) return;
    if (!it->second.rom) ::close(it->second.host_fd);
    files.erase(it);
}
NATIVE("_ZN2nn2fs11GetFileSizeEPlNS0_10FileHandleE") {
    OpenFile* f = FindFile(c->x[1]);
    if (!f) return Ret(c, ResultPathNotFound);
    if (f->rom) {
        *Ptr<s64>(c, 0) = (s64)f->rom_size;
    } else {
        struct stat st;
        fstat(f->host_fd, &st);
        *Ptr<s64>(c, 0) = st.st_size;
    }
    Ret(c, 0);
}
NATIVE("_ZN2nn2fs11SetFileSizeENS0_10FileHandleEl") {
    OpenFile* f = FindFile(c->x[0]);
    if (!f || f->rom) return Ret(c, ResultUnsupported);
    Ret(c, ftruncate(f->host_fd, (off_t)c->x[1]) == 0 ? 0 : 0x7802);
}
NATIVE("_ZN2nn2fs10CreateFileEPKcl") {
    Mount m;
    std::string path;
    if (!Split(Ptr<const char>(c, 0), m, path) || m.rom) return Ret(c, ResultPathNotFound);
    const fs::path host = HostPath(m, path);
    if (fs::exists(host)) return Ret(c, ResultPathAlreadyExists);
    const int fd = ::open(host.string().c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
    if (fd < 0) return Ret(c, ResultPathNotFound);
    ftruncate(fd, (off_t)c->x[1]);
    ::close(fd);
    Ret(c, 0);
}
NATIVE("_ZN2nn2fs10DeleteFileEPKc") {
    Mount m;
    std::string path;
    if (!Split(Ptr<const char>(c, 0), m, path) || m.rom) return Ret(c, ResultPathNotFound);
    std::error_code ec;
    Ret(c, fs::remove(HostPath(m, path), ec) ? 0 : ResultPathNotFound);
}

// ---- directories ---------------------------------------------------------------------------------
NATIVE("_ZN2nn2fs12GetEntryTypeEPNS0_18DirectoryEntryTypeEPKc") {
    Mount m;
    std::string path;
    if (!Split(Ptr<const char>(c, 1), m, path)) return Ret(c, ResultPathNotFound);
    int type;
    if (m.rom) {
        const RomEntry* e = Rom().Find(path);
        if (!e) return Ret(c, ResultPathNotFound);
        type = e->directory ? 0 : 1;
    } else {
        std::error_code ec;
        const auto st = fs::status(HostPath(m, path), ec);
        if (!fs::exists(st)) return Ret(c, ResultPathNotFound);
        type = fs::is_directory(st) ? 0 : 1;
    }
    *Ptr<u32>(c, 0) = (u32)type;
    Ret(c, 0);
}
NATIVE("_ZN2nn2fs15CreateDirectoryEPKc") {
    Mount m;
    std::string path;
    if (!Split(Ptr<const char>(c, 0), m, path) || m.rom) return Ret(c, ResultPathNotFound);
    std::error_code ec;
    if (fs::exists(HostPath(m, path))) return Ret(c, ResultPathAlreadyExists);
    Ret(c, fs::create_directory(HostPath(m, path), ec) ? 0 : ResultPathNotFound);
}
NATIVE("_ZN2nn2fs26DeleteDirectoryRecursivelyEPKc") {
    Mount m;
    std::string path;
    if (!Split(Ptr<const char>(c, 0), m, path) || m.rom) return Ret(c, ResultPathNotFound);
    std::error_code ec;
    fs::remove_all(HostPath(m, path), ec);
    Ret(c, 0);
}
NATIVE("_ZN2nn2fs13OpenDirectoryEPNS0_15DirectoryHandleEPKci") {
    Mount m;
    std::string path;
    if (!Split(Ptr<const char>(c, 1), m, path)) return Ret(c, ResultPathNotFound);
    const int mode = (int)c->x[2]; // 1 directories, 2 files
    OpenDir d;
    if (m.rom) {
        const RomEntry* e = Rom().Find(path);
        if (!e || !e->directory) return Ret(c, ResultPathNotFound);
        for (const auto& [name, full] : e->children) {
            const RomEntry* child = Rom().Find(full);
            if (child->directory ? (mode & 1) : (mode & 2)) d.items.push_back({name, {child->directory, child->size}});
        }
    } else {
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(HostPath(m, path), ec)) {
            const bool dir = entry.is_directory();
            if (dir ? (mode & 1) : (mode & 2))
                d.items.push_back({entry.path().filename().string(), {dir, dir ? 0 : (u64)entry.file_size()}});
        }
        if (ec) return Ret(c, ResultPathNotFound);
    }
    const u64 h = next_handle++;
    {
        std::lock_guard lock{fs_mutex};
        directories.emplace(h, std::move(d));
    }
    *Ptr<u64>(c, 0) = h;
    Ret(c, 0);
}
// DirectoryEntry: char name[0x301]; u8 attributes; u8 pad[2]; s8 type; u8 pad[3]; s64 size.
NATIVE("_ZN2nn2fs13ReadDirectoryEPlPNS0_14DirectoryEntryENS0_15DirectoryHandleEl") {
    std::lock_guard lock{fs_mutex};
    auto it = directories.find(c->x[2]);
    if (it == directories.end()) return Ret(c, ResultPathNotFound);
    OpenDir& d = it->second;
    u8* out = Ptr<u8>(c, 1);
    s64 count = 0;
    for (; count < (s64)c->x[3] && d.next < d.items.size(); ++count, ++d.next) {
        u8* e = out + count * 0x310;
        std::memset(e, 0, 0x310);
        const auto& [name, info] = d.items[d.next];
        std::memcpy(e, name.c_str(), std::min<size_t>(name.size(), 0x300));
        e[0x304] = info.first ? 0 : 1;
        const s64 size = (s64)info.second;
        std::memcpy(e + 0x308, &size, 8);
    }
    *Ptr<s64>(c, 0) = count;
    Ret(c, 0);
}
NATIVE("_ZN2nn2fs22GetDirectoryEntryCountEPlNS0_15DirectoryHandleE") {
    std::lock_guard lock{fs_mutex};
    auto it = directories.find(c->x[1]);
    if (it == directories.end()) return Ret(c, ResultPathNotFound);
    *Ptr<s64>(c, 0) = (s64)it->second.items.size();
    Ret(c, 0);
}
NATIVE("_ZN2nn2fs14CloseDirectoryENS0_15DirectoryHandleE") {
    std::lock_guard lock{fs_mutex};
    directories.erase(c->x[0]);
}

// A RomFS file's bytes (the host side's own reads: fonts and UI assets for the graphics menu).
bool ReadRomFile(const std::string& path, std::vector<unsigned char>& out) {
    const RomEntry* e = Rom().Find(path);
    if (!e || e->directory) return false;
    out.resize(e->size);
    return pread(Rom().fd, out.data(), e->size, (off_t)e->offset) == (ssize_t)e->size;
}
// The save root (graphics settings are stored beside the save data).
std::string SettingsDirectory() { return DataRoot().string(); }
