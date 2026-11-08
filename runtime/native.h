// Native implementations of SDK functions. Each replaces the exported function of the same
// (mangled) name for every caller, recompiled code included. Arguments arrive in X0-X7 / V0-V7
// as on the console; results go back in X0 / V0.
#pragma once

#include "rt.h"
#include "kernel.h"

namespace rt {

struct NativeRegistration {
    NativeRegistration(const char* name, GuestFn fn);
    const char* name;
    GuestFn fn;
};
std::vector<NativeRegistration*>& NativeRegistrations();

template <typename T = u64> inline T Arg(Cpu* c, int i) { return (T)c->x[i]; }
template <typename T = void> inline T* Ptr(Cpu* c, int i) { return reinterpret_cast<T*>(c->x[i]); }
inline void Ret(Cpu* c, u64 v) { c->x[0] = v; }

// Calls an exported guest function (e.g. a recompiled SDK function) by its mangled name.
u64 CallExport(Cpu* c, const char* name, std::initializer_list<u64> args = {});
// Calls the recompiled implementation of an export that a native replaces.
u64 CallOriginal(Cpu* c, const char* name, std::initializer_list<u64> args = {});

} // namespace rt

#define NATIVE_CAT2(a, b) a##b
#define NATIVE_CAT(a, b) NATIVE_CAT2(a, b)
// NATIVE("mangled name") { body using Cpu* c }
#define NATIVE(mangled)                                                                          \
    static void NATIVE_CAT(native_fn_, __LINE__)(Cpu * c);                                       \
    static ::rt::NativeRegistration NATIVE_CAT(native_reg_, __LINE__){mangled, NATIVE_CAT(native_fn_, __LINE__)}; \
    static void NATIVE_CAT(native_fn_, __LINE__)([[maybe_unused]] Cpu * c)
