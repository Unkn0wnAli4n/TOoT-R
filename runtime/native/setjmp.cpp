// setjmp/longjmp across recompiled code. Guest setjmp returns a second time when the guest
// longjmps, which a C-call translation cannot express. The recompiler emits a host
// __builtin_setjmp at every setjmp call site (rt_setjmp_frame supplies its buffer and records the
// guest registers at the call); the native longjmp restores those registers, sets the result and
// resumes the recompiled caller right after its setjmp call.

#include <memory>
#include <unordered_map>

#include "../native.h"

using namespace rt;

namespace {

struct Frame {
    alignas(16) void* host[32]{}; // (__builtin_setjmp uses 5 words, a Windows jmp_buf 256 bytes)
    Cpu saved{};
};
thread_local std::unordered_map<u64, std::unique_ptr<Frame>> frames; // guest jmp_buf -> frame

[[noreturn]] void Resume(Cpu* c, u64 env, u64 value) {
    auto it = frames.find(env);
    if (it == frames.end()) Fatal("longjmp to a jmp_buf %#llx with no setjmp on this thread", (unsigned long long)env);
    Frame& f = *it->second;
    *c = f.saved; // registers as setjmp saw them (x30 is the call's return address)
    c->x[0] = value ? value : 1;
    RT_LONGJMP(f.host);
}

} // namespace

void** rt_setjmp_frame(Cpu* c) {
    auto& slot = frames[c->x[0]];
    if (!slot) slot = std::make_unique<Frame>();
    slot->saved = *c;
    return slot->host;
}

NATIVE("longjmp") { Resume(c, c->x[0], c->x[1]); }
NATIVE("_longjmp") { Resume(c, c->x[0], c->x[1]); }
