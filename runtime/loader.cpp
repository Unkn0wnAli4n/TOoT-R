// Loads the flat module images at their fixed addresses and links them (the job rtld does on the
// console): packed relative relocations, symbol relocations against the other modules' exports,
// with natively implemented SDK functions taking precedence.

#include <algorithm>
#include <fstream>
#include <iterator>
#include <unordered_map>

#include "native.h"
#include "rt.h"

extern const FunctionEntry main_functions[];
extern const size_t main_function_count;
extern const FunctionEntry sdk_functions[];
extern const size_t sdk_function_count;
extern const FunctionEntry subsdk0_functions[];
extern const size_t subsdk0_function_count;
extern const SlotEntry main_slots[], sdk_slots[], subsdk0_slots[];
extern const size_t main_slot_count, sdk_slot_count, subsdk0_slot_count;

namespace rt {
namespace {

enum : s64 {
    DT_NULL = 0, DT_PLTRELSZ = 2, DT_STRTAB = 5, DT_SYMTAB = 6, DT_RELA = 7, DT_RELASZ = 8,
    DT_INIT = 12, DT_JMPREL = 23, DT_INIT_ARRAY = 25, DT_INIT_ARRAYSZ = 27,
    DT_RELRSZ = 35, DT_RELR = 36,
};
enum : u32 { R_ABS64 = 257, R_GLOB_DAT = 1025, R_JUMP_SLOT = 1026, R_RELATIVE = 1027 };

struct Elf64Sym {
    u32 name;
    u8 info, other;
    u16 shndx;
    u64 value, size;
};

struct Dynamic {
    std::unordered_map<s64, u64> tags;
    u64 Get(s64 tag) const { auto it = tags.find(tag); return it == tags.end() ? 0 : it->second; }
};

Dynamic ReadDynamic(const Module& m) {
    const u8* img = reinterpret_cast<const u8*>(m.base);
    u32 mod0;
    std::memcpy(&mod0, img + 4, 4);
    s32 dyn_off;
    std::memcpy(&dyn_off, img + mod0 + 4, 4);
    Dynamic d;
    for (const u8* p = img + mod0 + dyn_off;; p += 16) {
        s64 tag;
        u64 value;
        std::memcpy(&tag, p, 8);
        std::memcpy(&value, p + 8, 8);
        if (tag == DT_NULL) break;
        d.tags.emplace(tag, value);
    }
    return d;
}

const char* SymbolName(const Module& m, const Dynamic& d, u32 index) {
    const auto* sym = reinterpret_cast<const Elf64Sym*>(m.base + d.Get(DT_SYMTAB)) + index;
    return reinterpret_cast<const char*>(m.base + d.Get(DT_STRTAB) + sym->name);
}

void ReadExports(Module& m, const Dynamic& d) {
    const u64 symtab = d.Get(DT_SYMTAB), strtab = d.Get(DT_STRTAB);
    // The symbol table directly precedes the string table in these modules.
    for (u64 i = 1; symtab + (i + 1) * sizeof(Elf64Sym) <= strtab; ++i) {
        const auto* sym = reinterpret_cast<const Elf64Sym*>(m.base + symtab) + i;
        if (!sym->value || !sym->shndx) continue;
        m.exports.push_back({reinterpret_cast<const char*>(m.base + strtab + sym->name), m.base + sym->value,
                             sym->size, (sym->info & 0xf) == 2});
    }
}

std::unordered_map<std::string_view, u64>& GlobalSymbols() {
    static std::unordered_map<std::string_view, u64> symbols;
    return symbols;
}

u64 Resolve(std::string_view name) {
    auto& symbols = GlobalSymbols();
    if (auto it = symbols.find(name); it != symbols.end()) return it->second;
    if (FindNative(name)) return RegisterNative(name, nullptr); // native-only symbol
    return 0;
}

void Relocate(Module& m, const Dynamic& d) {
    // Packed relative relocations.
    const u64 relr = d.Get(DT_RELR), relr_size = d.Get(DT_RELRSZ);
    u64 where = 0;
    for (u64 off = 0; off < relr_size; off += 8) {
        u64 entry;
        std::memcpy(&entry, reinterpret_cast<void*>(m.base + relr + off), 8);
        if ((entry & 1) == 0) {
            where = m.base + entry;
            *reinterpret_cast<u64*>(where) += m.base;
            where += 8;
        } else {
            for (int i = 0; (entry >>= 1) != 0; ++i)
                if (entry & 1) *reinterpret_cast<u64*>(where + i * 8) += m.base;
            where += 63 * 8;
        }
    }
    size_t unresolved = 0;
    const auto apply = [&](u64 start, u64 size) {
        for (u64 off = 0; off < size; off += 24) {
            u64 r_offset, r_info;
            s64 r_addend;
            std::memcpy(&r_offset, reinterpret_cast<void*>(m.base + start + off), 8);
            std::memcpy(&r_info, reinterpret_cast<void*>(m.base + start + off + 8), 8);
            std::memcpy(&r_addend, reinterpret_cast<void*>(m.base + start + off + 16), 8);
            auto* target = reinterpret_cast<u64*>(m.base + r_offset);
            const u32 type = static_cast<u32>(r_info);
            const u32 sym = static_cast<u32>(r_info >> 32);
            switch (type) {
            case R_RELATIVE:
                *target = m.base + r_addend;
                break;
            case R_ABS64: case R_GLOB_DAT: case R_JUMP_SLOT: {
                const char* name = SymbolName(m, d, sym);
                const u64 value = Resolve(name);
                if (!value) {
                    ++unresolved;
                    if (unresolved <= 20) RT_LOG("%s: unresolved symbol %s", m.name.c_str(), name);
                }
                *target = value + (type == R_JUMP_SLOT ? 0 : r_addend);
                break;
            }
            default:
                Fatal("%s: unsupported relocation type %u", m.name.c_str(), type);
            }
        }
    };
    apply(d.Get(DT_RELA), d.Get(DT_RELASZ));
    apply(d.Get(DT_JMPREL), d.Get(DT_PLTRELSZ));
    if (unresolved) RT_LOG("%s: %zu unresolved symbol relocations", m.name.c_str(), unresolved);
}

} // namespace

std::vector<Module>& Modules() {
    static std::vector<Module> modules;
    return modules;
}

void LoadModules(const std::string& data_dir) {
    struct Spec { const char* name; u64 base; const FunctionEntry* fns; size_t count; const SlotEntry* slots; size_t slot_count; };
    // Symbol lookup order matches the console's module order: main, subsdk0, sdk.
    const Spec specs[] = {
        {"main", MainBase, main_functions, main_function_count, main_slots, main_slot_count},
        {"subsdk0", SubsdkBase, subsdk0_functions, subsdk0_function_count, subsdk0_slots, subsdk0_slot_count},
        {"sdk", SdkBase, sdk_functions, sdk_function_count, sdk_slots, sdk_slot_count},
    };
    auto& modules = Modules();
    for (const auto& spec : specs) {
        const std::string path = data_dir + "/" + spec.name + ".img";
        std::ifstream f(path, std::ios::binary);
        if (!f) Fatal("missing %s (see extraction tools and build instructions in STATUS.md)", path.c_str());
        std::vector<char> data((std::istreambuf_iterator<char>(f)), {});
        Module m;
        m.name = spec.name;
        m.base = spec.base;
        m.size = data.size();
        MapFixed(m.base, m.size);
        std::memcpy(reinterpret_cast<void*>(m.base), data.data(), data.size());
        m.functions = spec.fns;
        m.function_count = spec.count;
        m.slots = spec.slots;
        m.slot_count = spec.slot_count;
        for (size_t i = 0; i < spec.count; ++i) RegisterFunction(spec.fns[i].address, spec.fns[i].fn);
        modules.push_back(std::move(m));
        RT_LOG("loaded %s at %#llx (%zu functions)", spec.name, (unsigned long long)spec.base, spec.count);
    }
    std::vector<Dynamic> dyns;
    for (auto& m : modules) {
        dyns.push_back(ReadDynamic(m));
        ReadExports(m, dyns.back());
        for (const auto& e : m.exports) GlobalSymbols().emplace(e.name, e.address);
        const Dynamic& d = dyns.back();
        m.init = d.Get(DT_INIT) ? m.base + d.Get(DT_INIT) : 0;
        m.init_array = d.Get(DT_INIT_ARRAY) ? m.base + d.Get(DT_INIT_ARRAY) : 0;
        m.init_array_size = d.Get(DT_INIT_ARRAYSZ);
    }
    for (size_t i = 0; i < modules.size(); ++i) Relocate(modules[i], dyns[i]);
}

// A native implementation replaces the exported function at its own address, so module imports
// and calls from recompiled code (generated as dispatched calls) both reach it.
void ApplyNatives() {
    size_t applied = 0;
    for (auto* r : NativeRegistrations()) {
        const auto it = GlobalSymbols().find(r->name);
        if (it == GlobalSymbols().end()) continue;
        RegisterFunction(it->second, r->fn);
        // Direct calls inside the defining module go through its entry slot.
        for (const auto& m : Modules())
            for (size_t i = 0; i < m.slot_count; ++i)
                if (m.slots[i].address == it->second) *m.slots[i].slot = r->fn;
        ++applied;
    }
    RT_LOG("%zu native SDK functions installed", applied);
}

// Calls the recompiled implementation of an export, even when a native replaces it.
u64 CallOriginal(Cpu* c, const char* name, std::initializer_list<u64> args) {
    const auto it = GlobalSymbols().find(name);
    if (it == GlobalSymbols().end()) Fatal("export %s not found", name);
    for (const auto& m : Modules()) {
        const FunctionEntry* end = m.functions + m.function_count;
        const FunctionEntry* f = std::lower_bound(m.functions, end, it->second,
                                                  [](const FunctionEntry& e, u64 a) { return e.address < a; });
        if (f != end && f->address == it->second) {
            int i = 0;
            for (u64 a : args) c->x[i++] = a;
            const u64 lr = c->x[30];
            f->fn(c);
            c->x[30] = lr;
            return c->x[0];
        }
    }
    Fatal("no recompiled code for %s", name);
}

u64 CallExport(Cpu* c, const char* name, std::initializer_list<u64> args) {
    const auto it = GlobalSymbols().find(name);
    if (it == GlobalSymbols().end()) Fatal("export %s not found", name);
    return CallGuest(c, it->second, args);
}

std::string DescribeAddress(u64 address) {
    for (const auto& m : Modules()) {
        if (address < m.base || address >= m.base + m.size) continue;
        const Symbol* best = nullptr;
        for (const auto& s : m.exports)
            if (s.function && s.address <= address && (!best || s.address > best->address)) best = &s;
        char buf[512];
        if (best && address - best->address < 0x100000)
            std::snprintf(buf, sizeof buf, "%s+%#llx (%s+%#llx)", m.name.c_str(), (unsigned long long)(address - m.base),
                          best->name.c_str(), (unsigned long long)(address - best->address));
        else
            std::snprintf(buf, sizeof buf, "%s+%#llx", m.name.c_str(), (unsigned long long)(address - m.base));
        return buf;
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%#llx", (unsigned long long)address);
    return buf;
}

} // namespace rt
