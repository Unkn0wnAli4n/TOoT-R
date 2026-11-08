// Runtime internals shared by the loader, dispatcher, threads and native SDK implementations.
#pragma once

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "ops.h"
#include "platform.h"

namespace rt {

// Fixed guest (= host) load addresses, all below 2^39 as on the console.
constexpr u64 MainBase = 0x800000000ULL;
constexpr u64 SubsdkBase = 0x810000000ULL;
constexpr u64 SdkBase = 0x820000000ULL;
constexpr u64 NativeBase = 0x830000000ULL;  // addresses standing for native functions
constexpr u64 HeapBase = 0x1000000000ULL;   // nn::os memory heap
constexpr u64 HeapReserve = 0x400000000ULL; // 16 GiB reserved (address space only)
constexpr u64 ArenaBase = 0x1400000000ULL;  // stacks, TLS and other runtime allocations
constexpr u64 ArenaSize = 0x100000000ULL;
// Guest worker stacks alias heap pages here. Keep this separate from the runtime arena,
// whose allocations must not occupy the region the SDK searches for free stack addresses.
constexpr u64 StackBase = ArenaBase + ArenaSize;
constexpr u64 StackSize = 0x100000000ULL;

struct Symbol {
    std::string name;
    u64 address;
    u64 size;
    bool function;
};

struct Module {
    std::string name;
    u64 base = 0;
    u64 size = 0;
    std::vector<Symbol> exports;
    u64 init = 0;
    u64 init_array = 0, init_array_size = 0;
    const FunctionEntry* functions = nullptr;
    size_t function_count = 0;
    const SlotEntry* slots = nullptr;
    size_t slot_count = 0;
};

// Logging
#define RT_LOG(...) (std::fprintf(stderr, "[rt] " __VA_ARGS__), std::fputc('\n', stderr))
[[noreturn]] void Fatal(const char* fmt, ...);

// Memory (identity mapped guest memory)
void* MapFixed(u64 address, u64 size);
u64 ArenaAlloc(u64 size, u64 align = 16);

// Modules
std::vector<Module>& Modules();
void LoadModules(const std::string& data_dir);
std::string DescribeAddress(u64 address);

// Dispatch: guest address -> host function
void RegisterFunction(u64 address, GuestFn fn);
GuestFn LookupFunction(u64 address);
u64 RegisterNative(std::string_view name, GuestFn fn); // returns its guest address
GuestFn FindNative(std::string_view name);
std::string NativeName(u64 address);

// Calling guest code from native code with up to 8 integer arguments; returns X0.
u64 CallGuest(Cpu* c, u64 address, std::initializer_list<u64> args = {});

// Threads
Cpu* CurrentCpu();
Cpu* CreateCpu(u64 stack_top, u64 tls);
void BindCpu(Cpu* c);

} // namespace rt
