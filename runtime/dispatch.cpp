// Guest address -> host function dispatch, native function registry, guest calls and traps.

#include <chrono>
#include <xmmintrin.h>
#include <cinttypes>
#include <mutex>
#include <unordered_map>

#include "native.h"

// Exclusive-monitor granules (ops.h): lock + version per hashed 8-byte granule.
ExclusiveGranule exclusive_granules[1u << 16];
#include "rt.h"

namespace rt {
namespace {

// Open-addressing table, written during start-up and when natives are registered.
struct FunctionTable {
    std::vector<FunctionEntry> slots = std::vector<FunctionEntry>(1u << 20);
    size_t used = 0;

    static size_t Hash(u64 a) { return (size_t)((a >> 2) * 0x9E3779B97F4A7C15ULL >> 44); }

    void Insert(u64 address, GuestFn fn) {
        if (used * 2 > slots.size()) Fatal("function table full");
        for (size_t i = Hash(address);; i = (i + 1) & (slots.size() - 1)) {
            if (slots[i].address == address || slots[i].address == 0) {
                if (slots[i].address == 0) ++used;
                slots[i].fn = fn; // (an entry's address is published after its function)
                __atomic_store_n(&slots[i].address, address, __ATOMIC_RELEASE);
                return;
            }
        }
    }
    const FunctionEntry* FindEntry(u64 address) const {
        for (size_t i = Hash(address);; i = (i + 1) & (slots.size() - 1)) {
            if (slots[i].address == address) return &slots[i];
            if (slots[i].address == 0) return nullptr;
        }
    }
    GuestFn Find(u64 address) const { const FunctionEntry* e = FindEntry(address); return e ? e->fn : nullptr; }
};

FunctionTable& Table() {
    static FunctionTable table;
    return table;
}

struct NativeInfo {
    std::string name;
    GuestFn fn;
    u64 address;
};
std::mutex native_mutex;
std::unordered_map<std::string, NativeInfo>& NativeByName() {
    static std::unordered_map<std::string, NativeInfo> map;
    return map;
}

} // namespace

void RegisterFunction(u64 address, GuestFn fn) { Table().Insert(address, fn); }
GuestFn LookupFunction(u64 address) { return Table().Find(address); }

std::vector<NativeRegistration*>& NativeRegistrations() {
    static std::vector<NativeRegistration*> list;
    return list;
}

NativeRegistration::NativeRegistration(const char* name_, GuestFn fn_) : name(name_), fn(fn_) {
    NativeRegistrations().push_back(this);
}

GuestFn FindNative(std::string_view name) {
    for (auto* r : NativeRegistrations())
        if (name == r->name) return r->fn;
    return nullptr;
}

u64 RegisterNative(std::string_view name, GuestFn fn) {
    std::lock_guard lock{native_mutex};
    auto& map = NativeByName();
    const std::string key{name};
    if (auto it = map.find(key); it != map.end()) return it->second.address;
    if (!fn) fn = FindNative(name);
    if (!fn) Fatal("no native implementation for %s", key.c_str());
    if (FindNative(name)) fn = FindNative(name);
    const u64 address = NativeBase + map.size() * 16;
    map.emplace(key, NativeInfo{key, fn, address});
    RegisterFunction(address, fn);
    return address;
}

std::string NativeName(u64 address) {
    std::lock_guard lock{native_mutex};
    for (const auto& [name, info] : NativeByName())
        if (info.address == address) return name;
    return {};
}

u64 CallGuest(Cpu* c, u64 address, std::initializer_list<u64> args) {
    int i = 0;
    for (u64 a : args) c->x[i++] = a;
    const GuestFn fn = LookupFunction(address);
    if (!fn) Fatal("call to unknown guest function %s", DescribeAddress(address).c_str());
    c->pc = address;
    const u64 saved_lr = c->x[30];
    fn(c);
    c->x[30] = saved_lr;
    return c->x[0];
}

} // namespace rt

using namespace rt;

namespace rt {
// Optional call tracer (set by subsystems for debugging): called for every dispatched call.
void (*g_call_tracer)(Cpu* c, u64 target) = nullptr;
}

const FunctionEntry* rt_call_lookup(Cpu* c, u64 target, const FunctionEntry** cache) {
    if (rt::g_call_tracer) { (rt_call)(c, target); return nullptr; } // every call is traced
    const FunctionEntry* e = Table().FindEntry(target);
    if (!e) { (rt_call)(c, target); return nullptr; } // (reports the unknown target)
    __atomic_store_n(cache, e, __ATOMIC_RELAXED);
    return e;
}

void (rt_call)(Cpu* c, u64 target) {
    c->pc = target; // lets shared native stubs identify the function that was called
    if (g_call_tracer) g_call_tracer(c, target);
    const GuestFn fn = LookupFunction(target);
    if (!fn) Fatal("indirect branch to unknown address %s (lr %s)", DescribeAddress(target).c_str(),
                   DescribeAddress(c->x[30]).c_str());
    fn(c);
}

void rt_unreachable(Cpu* c, u64 function) {
    Fatal("unreachable code in %s (lr %s)", DescribeAddress(function).c_str(), DescribeAddress(c->x[30]).c_str());
}

void rt_exception(Cpu* c, u64 pc, u64 exception) {
    u32 word = 0;
    std::memcpy(&word, reinterpret_cast<void*>(pc), 4);
    if ((word & 0xffe0001fu) == 0xd4200000u) // BRK: the game's abort path
        Fatal("guest breakpoint #%u at %s (lr %s)", (word >> 5) & 0xffff, DescribeAddress(pc).c_str(),
              DescribeAddress(c->x[30]).c_str());
    Fatal("guest exception %llu at %s, instruction %08x (lr %s)", (unsigned long long)exception,
          DescribeAddress(pc).c_str(), word, DescribeAddress(c->x[30]).c_str());
}


void rt_set_fpcr(Cpu* c, u32 value) {
    // FZ (bit 24) maps to MXCSR flush-to-zero and denormals-are-zero; rounding stays nearest.
    if ((value ^ c->fpcr) & 0x00c00000u) RT_LOG("guest set FPCR rounding mode %u (not applied)", (value >> 22) & 3);
    u32 csr = _mm_getcsr() & ~0x8040u;
    if (value & 0x01000000u) csr |= 0x8040u;
    _mm_setcsr(csr);
    c->fpcr = value;
}

u64 rt_cntpct() {
    // The console's system counter runs at 19.2 MHz.
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return (u64)((unsigned __int128)ns * 192 / 10000);
}

u32 rt_crc32_iso(u32 crc, u64 value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        crc ^= (u8)(value >> (8 * i));
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1)));
    }
    return crc;
}
