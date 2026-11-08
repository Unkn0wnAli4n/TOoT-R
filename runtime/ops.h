// Semantics of the lifted AArch64 operations used by the recompiled code. Every op_* function
// implements one dynarmic IR opcode of the same name; flags_* compute NZCV (PSTATE layout, N in
// bit 31) for flag-setting forms. Guest addresses are host addresses (identity mapping).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <nmmintrin.h>

// Host setjmp for guest setjmp call sites (see native/setjmp.cpp; the buffer holds 256 bytes).
// On Win64, LLVM's __builtin_setjmp does not restore the frame correctly; the CRT's _setjmp with no
// unwind frame saves every callee-saved register and longjmp just restores them.
#ifdef _WIN32
#include <setjmp.h>
#define RT_SETJMP(buffer) _setjmp(reinterpret_cast<_JBTYPE*>(buffer), nullptr)
#define RT_LONGJMP(buffer) longjmp(reinterpret_cast<_JBTYPE*>(buffer), 1)
#else
#define RT_SETJMP(buffer) __builtin_setjmp(buffer)
#define RT_LONGJMP(buffer) __builtin_longjmp(buffer, 1)
#endif

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;
using u128 = unsigned __int128;

typedef u8 V128 __attribute__((vector_size(16), aligned(16)));
typedef u8 u8x16 __attribute__((vector_size(16)));
typedef u16 u16x8 __attribute__((vector_size(16)));
typedef u32 u32x4 __attribute__((vector_size(16)));
typedef u64 u64x2 __attribute__((vector_size(16)));
typedef s8 s8x16 __attribute__((vector_size(16)));
typedef s16 s16x8 __attribute__((vector_size(16)));
typedef s32 s32x4 __attribute__((vector_size(16)));
typedef s64 s64x2 __attribute__((vector_size(16)));
typedef float f32x4 __attribute__((vector_size(16)));
typedef double f64x2 __attribute__((vector_size(16)));

struct alignas(16) Cpu {
    V128 v[32];
    u64 x[32]; // x[31] unused (SP is separate)
    u64 sp;
    u64 pc;
    u64 tpidr;
    u64 tpidrro;
    u32 nzcv;
    u32 fpcr;
    u32 fpsr;
    bool excl_valid;
    u64 excl_addr;
    u128 excl_value;
    u32 excl_version; // monitor version seen by the exclusive load (see ExclusiveStore)
    void* thread; // runtime thread object
};

using GuestFn = void (*)(Cpu*);
struct FunctionEntry {
    u64 address;
    GuestFn fn;
};

// Runtime services (runtime/*.cpp).
void rt_call(Cpu* c, u64 target);
[[noreturn]] void rt_unreachable(Cpu* c, u64 function);
void rt_exception(Cpu* c, u64 pc, u64 exception);
void rt_svc(Cpu* c, u32 imm);
void rt_set_fpcr(Cpu* c, u32 value);
u64 rt_cntpct();

#define OP static inline __attribute__((always_inline))

// Indirect calls: each call site remembers the function-table entry of its last target (entries
// never move, and an entry's function is replaced in place), so a repeated target skips the lookup.
const FunctionEntry* rt_call_lookup(Cpu* c, u64 target, const FunctionEntry** cache);
OP void rt_call_cached(Cpu* c, u64 target, const FunctionEntry** cache) {
    const FunctionEntry* e = __atomic_load_n(cache, __ATOMIC_RELAXED);
    if (__builtin_expect(!e || e->address != target, 0)) { e = rt_call_lookup(c, target, cache); if (!e) return; }
    c->pc = target; // lets shared native stubs identify the function that was called
    e->fn(c);
}
#define rt_call(c, target) do { static const FunctionEntry* rt_call_site_; rt_call_cached((c), (target), &rt_call_site_); } while (0)

// ---- condition codes -------------------------------------------------------------------------
OP bool rt_cond(u32 nzcv, int cond) {
    const bool n = nzcv >> 31 & 1, z = nzcv >> 30 & 1, cf = nzcv >> 29 & 1, v = nzcv >> 28 & 1;
    bool r;
    switch (cond >> 1) {
    case 0: r = z; break;
    case 1: r = cf; break;
    case 2: r = n; break;
    case 3: r = v; break;
    case 4: r = cf && !z; break;
    case 5: r = n == v; break;
    case 6: r = n == v && !z; break;
    default: return true; // AL, NV
    }
    return (cond & 1) ? !r : r;
}

// ---- bit casts -------------------------------------------------------------------------------
OP float F32(u32 v) { float f; std::memcpy(&f, &v, 4); return f; }
OP double F64(u64 v) { double f; std::memcpy(&f, &v, 8); return f; }
OP u32 B32(float f) { u32 v; std::memcpy(&v, &f, 4); return v; }
OP u64 B64(double f) { u64 v; std::memcpy(&v, &f, 8); return v; }
OP V128 Q(u64 lo, u64 hi) { u64x2 r = {lo, hi}; return (V128)r; }
OP u64 Lo(V128 v) { return ((u64x2)v)[0]; }
OP u64 Hi(V128 v) { return ((u64x2)v)[1]; }

// ---- register views --------------------------------------------------------------------------
OP V128 op_GetS(V128 v) { u32x4 r = {((u32x4)v)[0], 0, 0, 0}; return (V128)r; }
OP V128 op_GetD(V128 v) { return Q(Lo(v), 0); }

// ---- memory ----------------------------------------------------------------------------------
// AccType: 5 ORDERED, 6 ORDEREDRW, 7 LIMITEDORDERED use acquire/release ordering.
OP bool Ordered(int acc) { return acc >= 5 && acc <= 7; }
template <typename T> OP T Load(u64 addr, int acc) {
    T v;
    if (Ordered(acc) && (addr % sizeof(T)) == 0 && sizeof(T) <= 8) {
        v = __atomic_load_n(reinterpret_cast<T*>(addr), __ATOMIC_ACQUIRE);
    } else {
        std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(T));
    }
    return v;
}
template <typename T> OP void Store(u64 addr, T v, int acc) {
    if (Ordered(acc) && (addr % sizeof(T)) == 0 && sizeof(T) <= 8) {
        __atomic_store_n(reinterpret_cast<T*>(addr), v, __ATOMIC_RELEASE);
    } else {
        std::memcpy(reinterpret_cast<void*>(addr), &v, sizeof(T));
    }
}
OP u8 op_ReadMemory8(u64 a, int acc) { return Load<u8>(a, acc); }
OP u16 op_ReadMemory16(u64 a, int acc) { return Load<u16>(a, acc); }
OP u32 op_ReadMemory32(u64 a, int acc) { return Load<u32>(a, acc); }
OP u64 op_ReadMemory64(u64 a, int acc) { return Load<u64>(a, acc); }
OP V128 op_ReadMemory128(u64 a, int acc) { V128 v; std::memcpy(&v, reinterpret_cast<const void*>(a), 16); (void)acc; return v; }
OP void op_WriteMemory8(u64 a, u8 v, int acc) { Store<u8>(a, v, acc); }
OP void op_WriteMemory16(u64 a, u16 v, int acc) { Store<u16>(a, v, acc); }
OP void op_WriteMemory32(u64 a, u32 v, int acc) { Store<u32>(a, v, acc); }
OP void op_WriteMemory64(u64 a, u64 v, int acc) { Store<u64>(a, v, acc); }
OP void op_WriteMemory128(u64 a, V128 v, int acc) { std::memcpy(reinterpret_cast<void*>(a), &v, 16); (void)acc; }

// Exclusive monitor. A store succeeds only if the location still holds the loaded value AND no
// other exclusive store hit it in between (a per-granule version, as the hardware monitor is
// cleared by any intervening write): a value-only compare-and-swap lets lock-free stacks and
// queues succeed on a stale pointer (ABA), corrupting them. Returns 0 on success, 1 on failure.
struct ExclusiveGranule { u32 lock; u32 version; };
extern ExclusiveGranule exclusive_granules[1u << 16];
OP ExclusiveGranule& GranuleFor(u64 a) { return exclusive_granules[(a >> 3) & ((1u << 16) - 1)]; }
OP void LockGranule(ExclusiveGranule& g) {
    while (__atomic_exchange_n(&g.lock, 1u, __ATOMIC_ACQUIRE)) while (__atomic_load_n(&g.lock, __ATOMIC_RELAXED)) __builtin_ia32_pause();
}
OP void UnlockGranule(ExclusiveGranule& g) { __atomic_store_n(&g.lock, 0u, __ATOMIC_RELEASE); }
template <typename T> OP T ExclusiveLoad(Cpu* c, u64 a, int acc) {
    c->excl_version = __atomic_load_n(&GranuleFor(a).version, __ATOMIC_ACQUIRE);
    T v = __atomic_load_n(reinterpret_cast<T*>(a), Ordered(acc) ? __ATOMIC_ACQUIRE : __ATOMIC_RELAXED);
    c->excl_valid = true;
    c->excl_addr = a;
    c->excl_value = v;
    return v;
}
template <typename T> OP u32 ExclusiveStore(Cpu* c, u64 a, T v, int acc) {
    if (!c->excl_valid || c->excl_addr != a) { c->excl_valid = false; return 1; }
    c->excl_valid = false;
    auto& g = GranuleFor(a);
    LockGranule(g);
    T expected = static_cast<T>(c->excl_value);
    const bool ok = g.version == c->excl_version &&
        __atomic_compare_exchange_n(reinterpret_cast<T*>(a), &expected, v, false, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
    if (ok) __atomic_store_n(&g.version, g.version + 1, __ATOMIC_RELEASE);
    UnlockGranule(g);
    (void)acc;
    return ok ? 0 : 1;
}
OP u8 op_ExclusiveReadMemory8(Cpu* c, u64 a, int acc) { return ExclusiveLoad<u8>(c, a, acc); }
OP u16 op_ExclusiveReadMemory16(Cpu* c, u64 a, int acc) { return ExclusiveLoad<u16>(c, a, acc); }
OP u32 op_ExclusiveReadMemory32(Cpu* c, u64 a, int acc) { return ExclusiveLoad<u32>(c, a, acc); }
OP u64 op_ExclusiveReadMemory64(Cpu* c, u64 a, int acc) { return ExclusiveLoad<u64>(c, a, acc); }
OP u32 op_ExclusiveWriteMemory8(Cpu* c, u64 a, u8 v, int acc) { return ExclusiveStore<u8>(c, a, v, acc); }
OP u32 op_ExclusiveWriteMemory16(Cpu* c, u64 a, u16 v, int acc) { return ExclusiveStore<u16>(c, a, v, acc); }
OP u32 op_ExclusiveWriteMemory32(Cpu* c, u64 a, u32 v, int acc) { return ExclusiveStore<u32>(c, a, v, acc); }
OP u32 op_ExclusiveWriteMemory64(Cpu* c, u64 a, u64 v, int acc) { return ExclusiveStore<u64>(c, a, v, acc); }
OP V128 op_ExclusiveReadMemory128(Cpu* c, u64 a, int) {
    c->excl_version = __atomic_load_n(&GranuleFor(a).version, __ATOMIC_ACQUIRE);
    u128 v = __sync_val_compare_and_swap(reinterpret_cast<u128*>(a), (u128)0, (u128)0);
    c->excl_valid = true;
    c->excl_addr = a;
    c->excl_value = v;
    return Q((u64)v, (u64)(v >> 64));
}
OP u32 op_ExclusiveWriteMemory128(Cpu* c, u64 a, V128 v, int) {
    if (!c->excl_valid || c->excl_addr != a) { c->excl_valid = false; return 1; }
    c->excl_valid = false;
    const u128 n = (u128)Lo(v) | ((u128)Hi(v) << 64);
    auto& g = GranuleFor(a);
    LockGranule(g);
    const bool ok = g.version == c->excl_version && __sync_bool_compare_and_swap(reinterpret_cast<u128*>(a), c->excl_value, n);
    if (ok) __atomic_store_n(&g.version, g.version + 1, __ATOMIC_RELEASE);
    UnlockGranule(g);
    return ok ? 0 : 1;
}

// ---- integer ---------------------------------------------------------------------------------
OP u32 op_Add32(u32 a, u32 b, bool cin) { return a + b + cin; }
OP u64 op_Add64(u64 a, u64 b, bool cin) { return a + b + cin; }
OP u32 op_Sub32(u32 a, u32 b, bool cin) { return a + ~b + cin; }
OP u64 op_Sub64(u64 a, u64 b, bool cin) { return a + ~b + cin; }
template <typename T, typename S> OP u32 AddFlags(T a, T b, bool cin) {
    const T r = a + b + cin;
    const bool n = (S)r < 0, z = r == 0;
    const bool cf = cin ? r <= a : r < a;
    const bool v = (S)((a ^ r) & (b ^ r)) < 0;
    return (u32)n << 31 | (u32)z << 30 | (u32)cf << 29 | (u32)v << 28;
}
OP u32 flags_Add32(u32 a, u32 b, bool cin) { return AddFlags<u32, s32>(a, b, cin); }
OP u32 flags_Add64(u64 a, u64 b, bool cin) { return AddFlags<u64, s64>(a, b, cin); }
OP u32 flags_Sub32(u32 a, u32 b, bool cin) { return AddFlags<u32, s32>(a, ~b, cin); }
OP u32 flags_Sub64(u64 a, u64 b, bool cin) { return AddFlags<u64, s64>(a, ~b, cin); }
OP u32 LogicFlags32(u32 r) { return (r >> 31) << 31 | (u32)(r == 0) << 30; }
OP u32 LogicFlags64(u64 r) { return (u32)(r >> 63) << 31 | (u32)(r == 0) << 30; }
OP u32 op_And32(u32 a, u32 b) { return a & b; }
OP u64 op_And64(u64 a, u64 b) { return a & b; }
OP u32 flags_And32(u32 a, u32 b) { return LogicFlags32(a & b); }
OP u32 flags_And64(u64 a, u64 b) { return LogicFlags64(a & b); }
OP u32 op_AndNot32(u32 a, u32 b) { return a & ~b; }
OP u64 op_AndNot64(u64 a, u64 b) { return a & ~b; }
OP u32 flags_AndNot32(u32 a, u32 b) { return LogicFlags32(a & ~b); }
OP u32 flags_AndNot64(u64 a, u64 b) { return LogicFlags64(a & ~b); }
OP u32 op_Or32(u32 a, u32 b) { return a | b; }
OP u64 op_Or64(u64 a, u64 b) { return a | b; }
OP u32 op_Eor32(u32 a, u32 b) { return a ^ b; }
OP u64 op_Eor64(u64 a, u64 b) { return a ^ b; }
OP u32 op_Not32(u32 a) { return ~a; }
OP u64 op_Not64(u64 a) { return ~a; }
OP u32 op_Mul32(u32 a, u32 b) { return a * b; }
OP u64 op_Mul64(u64 a, u64 b) { return a * b; }
OP u64 op_UnsignedMultiplyHigh64(u64 a, u64 b) { return (u64)(((u128)a * b) >> 64); }
OP u64 op_SignedMultiplyHigh64(u64 a, u64 b) { return (u64)(((__int128)(s64)a * (s64)b) >> 64); }
OP u32 op_UnsignedDiv32(u32 a, u32 b) { return b ? a / b : 0; }
OP u64 op_UnsignedDiv64(u64 a, u64 b) { return b ? a / b : 0; }
OP u32 op_SignedDiv32(u32 a, u32 b) {
    if (b == 0) return 0;
    if ((s32)a == INT32_MIN && (s32)b == -1) return a;
    return (u32)((s32)a / (s32)b);
}
OP u64 op_SignedDiv64(u64 a, u64 b) {
    if (b == 0) return 0;
    if ((s64)a == INT64_MIN && (s64)b == -1) return a;
    return (u64)((s64)a / (s64)b);
}
OP u32 op_MinSigned32(u32 a, u32 b) { return (s32)a < (s32)b ? a : b; }
OP u32 op_MaxSigned32(u32 a, u32 b) { return (s32)a > (s32)b ? a : b; }
OP u64 op_MinSigned64(u64 a, u64 b) { return (s64)a < (s64)b ? a : b; }
OP u64 op_MaxSigned64(u64 a, u64 b) { return (s64)a > (s64)b ? a : b; }
OP u32 op_MinUnsigned32(u32 a, u32 b) { return a < b ? a : b; }
OP u32 op_MaxUnsigned32(u32 a, u32 b) { return a > b ? a : b; }
OP u64 op_MinUnsigned64(u64 a, u64 b) { return a < b ? a : b; }
OP u64 op_MaxUnsigned64(u64 a, u64 b) { return a > b ? a : b; }

OP u32 op_LogicalShiftLeft32(u32 a, u8 s, bool) { return s >= 32 ? 0 : a << s; }
OP u32 op_LogicalShiftRight32(u32 a, u8 s, bool) { return s >= 32 ? 0 : a >> s; }
OP u32 op_ArithmeticShiftRight32(u32 a, u8 s, bool) { return (u32)((s32)a >> (s >= 32 ? 31 : s)); }
OP u32 op_BitRotateRight32(u32 a, u8 s, bool) { s &= 31; return s ? (a >> s) | (a << (32 - s)) : a; }
OP u64 op_LogicalShiftLeft64(u64 a, u8 s) { return s >= 64 ? 0 : a << s; }
OP u64 op_LogicalShiftRight64(u64 a, u8 s) { return s >= 64 ? 0 : a >> s; }
OP u64 op_ArithmeticShiftRight64(u64 a, u8 s) { return (u64)((s64)a >> (s >= 64 ? 63 : s)); }
OP u64 op_BitRotateRight64(u64 a, u8 s) { s &= 63; return s ? (a >> s) | (a << (64 - s)) : a; }
OP u32 op_LogicalShiftLeftMasked32(u32 a, u32 s) { return a << (s & 31); }
OP u32 op_LogicalShiftRightMasked32(u32 a, u32 s) { return a >> (s & 31); }
OP u32 op_ArithmeticShiftRightMasked32(u32 a, u32 s) { return (u32)((s32)a >> (s & 31)); }
OP u32 op_RotateRightMasked32(u32 a, u32 s) { return op_BitRotateRight32(a, (u8)(s & 31), false); }
OP u64 op_LogicalShiftLeftMasked64(u64 a, u64 s) { return a << (s & 63); }
OP u64 op_LogicalShiftRightMasked64(u64 a, u64 s) { return a >> (s & 63); }
OP u64 op_ArithmeticShiftRightMasked64(u64 a, u64 s) { return (u64)((s64)a >> (s & 63)); }
OP u64 op_RotateRightMasked64(u64 a, u64 s) { return op_BitRotateRight64(a, (u8)(s & 63)); }
OP u32 op_ExtractRegister32(u32 a, u32 b, u8 lsb) { return lsb ? (a >> lsb) | (b << (32 - lsb)) : a; }
OP u64 op_ExtractRegister64(u64 a, u64 b, u8 lsb) { return lsb ? (a >> lsb) | (b << (64 - lsb)) : a; }
OP u32 op_ReplicateBit32(u32 a, u8 bit) { return (a >> bit & 1) ? ~0u : 0u; }
OP u64 op_ReplicateBit64(u64 a, u8 bit) { return (a >> bit & 1) ? ~0ULL : 0ULL; }
OP bool op_TestBit(u64 a, u8 bit) { return a >> bit & 1; }
OP bool op_IsZero32(u32 a) { return a == 0; }
OP bool op_IsZero64(u64 a) { return a == 0; }
OP u32 op_CountLeadingZeros32(u32 a) { return a ? __builtin_clz(a) : 32; }
OP u64 op_CountLeadingZeros64(u64 a) { return a ? __builtin_clzll(a) : 64; }
OP u32 op_ByteReverseWord(u32 a) { return __builtin_bswap32(a); }
OP u16 op_ByteReverseHalf(u16 a) { return __builtin_bswap16(a); }
OP u64 op_ByteReverseDual(u64 a) { return __builtin_bswap64(a); }
OP u64 op_Pack2x32To1x64(u32 lo, u32 hi) { return (u64)lo | (u64)hi << 32; }
OP V128 op_Pack2x64To1x128(u64 lo, u64 hi) { return Q(lo, hi); }
OP u32 op_LeastSignificantWord(u64 a) { return (u32)a; }
OP u32 op_MostSignificantWord(u64 a) { return (u32)(a >> 32); }
OP u16 op_LeastSignificantHalf(u32 a) { return (u16)a; }
OP u8 op_LeastSignificantByte(u32 a) { return (u8)a; }
OP bool op_MostSignificantBit(u32 a) { return a >> 31; }
OP u32 op_ZeroExtendByteToWord(u8 a) { return a; }
OP u32 op_ZeroExtendHalfToWord(u16 a) { return a; }
OP u64 op_ZeroExtendByteToLong(u8 a) { return a; }
OP u64 op_ZeroExtendHalfToLong(u16 a) { return a; }
OP u64 op_ZeroExtendWordToLong(u32 a) { return a; }
OP V128 op_ZeroExtendLongToQuad(u64 a) { return Q(a, 0); }
OP u32 op_SignExtendByteToWord(u8 a) { return (u32)(s32)(s8)a; }
OP u32 op_SignExtendHalfToWord(u16 a) { return (u32)(s32)(s16)a; }
OP u64 op_SignExtendByteToLong(u8 a) { return (u64)(s64)(s8)a; }
OP u64 op_SignExtendHalfToLong(u16 a) { return (u64)(s64)(s16)a; }
OP u64 op_SignExtendWordToLong(u32 a) { return (u64)(s64)(s32)a; }
OP u32 op_NZCVFromPackedFlags(u32 a) { return a & 0xf0000000u; }

// CRC32: ISO polynomial (reflected 0xEDB88320) and Castagnoli (SSE4.2).
u32 rt_crc32_iso(u32 crc, u64 value, int bytes);
OP u32 op_CRC32ISO8(u32 crc, u32 v) { return rt_crc32_iso(crc, v, 1); }
OP u32 op_CRC32ISO16(u32 crc, u32 v) { return rt_crc32_iso(crc, v, 2); }
OP u32 op_CRC32ISO32(u32 crc, u32 v) { return rt_crc32_iso(crc, v, 4); }
OP u32 op_CRC32ISO64(u32 crc, u64 v) { return rt_crc32_iso(crc, v, 8); }
OP u32 op_CRC32Castagnoli8(u32 crc, u32 v) { return _mm_crc32_u8(crc, (u8)v); }
OP u32 op_CRC32Castagnoli16(u32 crc, u32 v) { return _mm_crc32_u16(crc, (u16)v); }
OP u32 op_CRC32Castagnoli32(u32 crc, u32 v) { return _mm_crc32_u32(crc, v); }
OP u32 op_CRC32Castagnoli64(u32 crc, u64 v) { return (u32)_mm_crc32_u64(crc, v); }

// ---- scalar floating point (FPCR: round to nearest, no flush to zero) --------------------------
template <typename F> OP bool IsSNaN(F f) {
    if constexpr (sizeof(F) == 4) { const u32 b = B32(f); return std::isnan(f) && !(b & 0x00400000u); }
    else { const u64 b = B64(f); return std::isnan(f) && !(b & 0x0008000000000000ULL); }
}
template <typename F> OP F Quiet(F f) {
    if constexpr (sizeof(F) == 4) return F32(B32(f) | 0x00400000u);
    else return F64(B64(f) | 0x0008000000000000ULL);
}
// AArch64 NaN propagation: signalling NaN first, then the first quiet NaN operand.
template <typename F> OP bool PickNaN(F a, F b, F& out) {
    if (IsSNaN(a)) { out = Quiet(a); return true; }
    if (IsSNaN(b)) { out = Quiet(b); return true; }
    if (std::isnan(a)) { out = a; return true; }
    if (std::isnan(b)) { out = b; return true; }
    return false;
}
template <typename F> OP F FMax(F a, F b) {
    F n; if (PickNaN(a, b, n)) return n;
    if (a == 0 && b == 0) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}
template <typename F> OP F FMin(F a, F b) {
    F n; if (PickNaN(a, b, n)) return n;
    if (a == 0 && b == 0) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
template <typename F> OP F FMaxNum(F a, F b) {
    if (std::isnan(a) && !IsSNaN(a) && !std::isnan(b)) a = -INFINITY;
    else if (std::isnan(b) && !IsSNaN(b) && !std::isnan(a)) b = -INFINITY;
    return FMax(a, b);
}
template <typename F> OP F FMinNum(F a, F b) {
    if (std::isnan(a) && !IsSNaN(a) && !std::isnan(b)) a = INFINITY;
    else if (std::isnan(b) && !IsSNaN(b) && !std::isnan(a)) b = INFINITY;
    return FMin(a, b);
}
template <typename F> OP u32 FCompare(F a, F b) {
    if (std::isnan(a) || std::isnan(b)) return 0x30000000u;
    if (a == b) return 0x60000000u;
    if (a < b) return 0x80000000u;
    return 0x20000000u;
}
// RoundingMode: 0 nearest-even, 1 +inf, 2 -inf, 3 zero, 4 nearest-away, 5 odd.
template <typename F> OP F RoundInt(F a, u8 mode) {
    switch (mode) {
    case 1: return std::ceil(a);
    case 2: return std::floor(a);
    case 3: return std::trunc(a);
    case 4: return std::round(a);
    default: return std::nearbyint(a);
    }
}
template <typename I, typename F> OP I ToFixed(F a, u8 fbits, u8 mode) {
    if (std::isnan(a)) return 0;
    // double is exact here: float/double inputs, power-of-two scaling and rounding; the bounds
    // (I's max/min as double) saturate correctly for 32- and 64-bit results.
    const double scaled = RoundInt<double>((double)a * std::ldexp(1.0, fbits), mode);
    if (scaled >= (double)std::numeric_limits<I>::max()) return std::numeric_limits<I>::max();
    if (scaled <= (double)std::numeric_limits<I>::min()) return std::numeric_limits<I>::min();
    return (I)scaled;
}
template <typename F, typename I> OP F FromFixed(I a, u8 fbits) {
    return fbits ? (F)((long double)a / std::ldexp(1.0L, fbits)) : (F)a;
}

OP u32 op_FPAdd32(u32 a, u32 b) { return B32(F32(a) + F32(b)); }
OP u64 op_FPAdd64(u64 a, u64 b) { return B64(F64(a) + F64(b)); }
OP u32 op_FPSub32(u32 a, u32 b) { return B32(F32(a) - F32(b)); }
OP u64 op_FPSub64(u64 a, u64 b) { return B64(F64(a) - F64(b)); }
OP u32 op_FPMul32(u32 a, u32 b) { return B32(F32(a) * F32(b)); }
OP u64 op_FPMul64(u64 a, u64 b) { return B64(F64(a) * F64(b)); }
OP u32 op_FPDiv32(u32 a, u32 b) { return B32(F32(a) / F32(b)); }
OP u64 op_FPDiv64(u64 a, u64 b) { return B64(F64(a) / F64(b)); }
OP u32 op_FPMulAdd32(u32 a, u32 b, u32 c) { return B32(std::fma(F32(b), F32(c), F32(a))); }
OP u64 op_FPMulAdd64(u64 a, u64 b, u64 c) { return B64(std::fma(F64(b), F64(c), F64(a))); }
OP u32 op_FPMulSub32(u32 a, u32 b, u32 c) { return B32(std::fma(-F32(b), F32(c), F32(a))); }
OP u64 op_FPMulSub64(u64 a, u64 b, u64 c) { return B64(std::fma(-F64(b), F64(c), F64(a))); }
OP u32 op_FPSqrt32(u32 a) { return B32(std::sqrt(F32(a))); }
OP u64 op_FPSqrt64(u64 a) { return B64(std::sqrt(F64(a))); }
OP u32 op_FPAbs32(u32 a) { return a & 0x7fffffffu; }
OP u64 op_FPAbs64(u64 a) { return a & 0x7fffffffffffffffULL; }
OP u32 op_FPNeg32(u32 a) { return a ^ 0x80000000u; }
OP u64 op_FPNeg64(u64 a) { return a ^ 0x8000000000000000ULL; }
OP u32 op_FPMax32(u32 a, u32 b) { return B32(FMax(F32(a), F32(b))); }
OP u64 op_FPMax64(u64 a, u64 b) { return B64(FMax(F64(a), F64(b))); }
OP u32 op_FPMin32(u32 a, u32 b) { return B32(FMin(F32(a), F32(b))); }
OP u64 op_FPMin64(u64 a, u64 b) { return B64(FMin(F64(a), F64(b))); }
OP u32 op_FPMaxNumeric32(u32 a, u32 b) { return B32(FMaxNum(F32(a), F32(b))); }
OP u64 op_FPMaxNumeric64(u64 a, u64 b) { return B64(FMaxNum(F64(a), F64(b))); }
OP u32 op_FPMinNumeric32(u32 a, u32 b) { return B32(FMinNum(F32(a), F32(b))); }
OP u64 op_FPMinNumeric64(u64 a, u64 b) { return B64(FMinNum(F64(a), F64(b))); }
OP u32 op_FPCompare32(u32 a, u32 b, bool) { return FCompare(F32(a), F32(b)); }
OP u32 op_FPCompare64(u64 a, u64 b, bool) { return FCompare(F64(a), F64(b)); }
OP u32 op_FPRoundInt32(u32 a, u8 mode, bool) { return B32(RoundInt(F32(a), mode)); }
OP u64 op_FPRoundInt64(u64 a, u8 mode, bool) { return B64(RoundInt(F64(a), mode)); }
OP u64 op_FPSingleToDouble(u32 a, u8) {
    const float f = F32(a);
    return std::isnan(f) ? B64((double)Quiet(f)) : B64((double)f);
}
OP u32 op_FPDoubleToSingle(u64 a, u8) {
    const double d = F64(a);
    return std::isnan(d) ? B32((float)Quiet(d)) : B32((float)d);
}
OP u32 op_FPSingleToFixedS32(u32 a, u8 f, u8 m) { return (u32)ToFixed<s32>(F32(a), f, m); }
OP u32 op_FPSingleToFixedU32(u32 a, u8 f, u8 m) { return ToFixed<u32>(F32(a), f, m); }
OP u64 op_FPSingleToFixedS64(u32 a, u8 f, u8 m) { return (u64)ToFixed<s64>(F32(a), f, m); }
OP u64 op_FPSingleToFixedU64(u32 a, u8 f, u8 m) { return ToFixed<u64>(F32(a), f, m); }
OP u32 op_FPDoubleToFixedS32(u64 a, u8 f, u8 m) { return (u32)ToFixed<s32>(F64(a), f, m); }
OP u32 op_FPDoubleToFixedU32(u64 a, u8 f, u8 m) { return ToFixed<u32>(F64(a), f, m); }
OP u64 op_FPDoubleToFixedS64(u64 a, u8 f, u8 m) { return (u64)ToFixed<s64>(F64(a), f, m); }
OP u64 op_FPDoubleToFixedU64(u64 a, u8 f, u8 m) { return ToFixed<u64>(F64(a), f, m); }
OP u32 op_FPFixedS32ToSingle(u32 a, u8 f, u8) { return B32(FromFixed<float>((s32)a, f)); }
OP u32 op_FPFixedU32ToSingle(u32 a, u8 f, u8) { return B32(FromFixed<float>(a, f)); }
OP u32 op_FPFixedS64ToSingle(u64 a, u8 f, u8) { return B32(FromFixed<float>((s64)a, f)); }
OP u32 op_FPFixedU64ToSingle(u64 a, u8 f, u8) { return B32(FromFixed<float>(a, f)); }
OP u64 op_FPFixedS32ToDouble(u32 a, u8 f, u8) { return B64(FromFixed<double>((s32)a, f)); }
OP u64 op_FPFixedU32ToDouble(u32 a, u8 f, u8) { return B64(FromFixed<double>(a, f)); }
OP u64 op_FPFixedS64ToDouble(u64 a, u8 f, u8) { return B64(FromFixed<double>((s64)a, f)); }
OP u64 op_FPFixedU64ToDouble(u64 a, u8 f, u8) { return B64(FromFixed<double>(a, f)); }

// ---- vectors: element access -----------------------------------------------------------------
OP V128 op_ZeroVector() { return Q(0, 0); }
OP V128 op_VectorZeroUpper(V128 a) { return Q(Lo(a), 0); }
OP u8 op_VectorGetElement8(V128 a, u8 i) { return ((u8x16)a)[i]; }
OP u16 op_VectorGetElement16(V128 a, u8 i) { return ((u16x8)a)[i]; }
OP u32 op_VectorGetElement32(V128 a, u8 i) { return ((u32x4)a)[i]; }
OP u64 op_VectorGetElement64(V128 a, u8 i) { return ((u64x2)a)[i]; }
OP V128 op_VectorSetElement8(V128 a, u8 i, u8 v) { u8x16 r = (u8x16)a; r[i] = v; return (V128)r; }
OP V128 op_VectorSetElement16(V128 a, u8 i, u16 v) { u16x8 r = (u16x8)a; r[i] = v; return (V128)r; }
OP V128 op_VectorSetElement32(V128 a, u8 i, u32 v) { u32x4 r = (u32x4)a; r[i] = v; return (V128)r; }
OP V128 op_VectorSetElement64(V128 a, u8 i, u64 v) { u64x2 r = (u64x2)a; r[i] = v; return (V128)r; }
OP V128 op_VectorBroadcast8(u8 v) { u8x16 r; for (int i = 0; i < 16; ++i) r[i] = v; return (V128)r; }
OP V128 op_VectorBroadcast16(u16 v) { u16x8 r; for (int i = 0; i < 8; ++i) r[i] = v; return (V128)r; }
OP V128 op_VectorBroadcast32(u32 v) { u32x4 r = {v, v, v, v}; return (V128)r; }
OP V128 op_VectorBroadcast64(u64 v) { return Q(v, v); }
OP V128 op_VectorBroadcastLower8(u8 v) { return op_VectorZeroUpper(op_VectorBroadcast8(v)); }
OP V128 op_VectorBroadcastLower16(u16 v) { return op_VectorZeroUpper(op_VectorBroadcast16(v)); }
OP V128 op_VectorBroadcastLower32(u32 v) { return op_VectorZeroUpper(op_VectorBroadcast32(v)); }
OP V128 op_VectorBroadcastElement8(V128 a, u8 i) { return op_VectorBroadcast8(((u8x16)a)[i]); }
OP V128 op_VectorBroadcastElement16(V128 a, u8 i) { return op_VectorBroadcast16(((u16x8)a)[i]); }
OP V128 op_VectorBroadcastElement32(V128 a, u8 i) { return op_VectorBroadcast32(((u32x4)a)[i]); }
OP V128 op_VectorBroadcastElement64(V128 a, u8 i) { return op_VectorBroadcast64(((u64x2)a)[i]); }
OP V128 op_VectorBroadcastElementLower8(V128 a, u8 i) { return op_VectorBroadcastLower8(((u8x16)a)[i]); }
OP V128 op_VectorBroadcastElementLower16(V128 a, u8 i) { return op_VectorBroadcastLower16(((u16x8)a)[i]); }
OP V128 op_VectorBroadcastElementLower32(V128 a, u8 i) { return op_VectorBroadcastLower32(((u32x4)a)[i]); }

// ---- vectors: logic --------------------------------------------------------------------------
OP V128 op_VectorAnd(V128 a, V128 b) { return a & b; }
OP V128 op_VectorAndNot(V128 a, V128 b) { return a & ~b; }
OP V128 op_VectorOr(V128 a, V128 b) { return a | b; }
OP V128 op_VectorEor(V128 a, V128 b) { return a ^ b; }
OP V128 op_VectorNot(V128 a) { return ~a; }

// ---- vectors: integer arithmetic -------------------------------------------------------------
#define LANEWISE2(NAME, VT, EXPR) \
    OP V128 op_##NAME(V128 a_, V128 b_) { VT a = (VT)a_, b = (VT)b_; return (V128)(EXPR); }
LANEWISE2(VectorAdd8, u8x16, a + b)
LANEWISE2(VectorAdd16, u16x8, a + b)
LANEWISE2(VectorAdd32, u32x4, a + b)
LANEWISE2(VectorAdd64, u64x2, a + b)
LANEWISE2(VectorSub8, u8x16, a - b)
LANEWISE2(VectorSub16, u16x8, a - b)
LANEWISE2(VectorSub32, u32x4, a - b)
LANEWISE2(VectorSub64, u64x2, a - b)
LANEWISE2(VectorMultiply16, u16x8, a * b)
LANEWISE2(VectorMultiply32, u32x4, a * b)
LANEWISE2(VectorEqual8, u8x16, (u8x16)(a == b))
LANEWISE2(VectorEqual16, u16x8, (u16x8)(a == b))
LANEWISE2(VectorEqual32, u32x4, (u32x4)(a == b))
LANEWISE2(VectorEqual64, u64x2, (u64x2)(a == b))
LANEWISE2(VectorGreaterS8, s8x16, (s8x16)(a > b))
LANEWISE2(VectorGreaterS16, s16x8, (s16x8)(a > b))
LANEWISE2(VectorGreaterS32, s32x4, (s32x4)(a > b))
LANEWISE2(VectorGreaterS64, s64x2, (s64x2)(a > b))
LANEWISE2(VectorMinS32, s32x4, a < b ? a : b)
LANEWISE2(VectorMaxS32, s32x4, a > b ? a : b)
LANEWISE2(VectorMinU8, u8x16, a < b ? a : b)
LANEWISE2(VectorMinU32, u32x4, a < b ? a : b)
LANEWISE2(VectorMinU64, u64x2, a < b ? a : b)
LANEWISE2(VectorMaxU32, u32x4, a > b ? a : b)

template <typename VT, typename T, int N> OP V128 SatSubU(V128 a_, V128 b_) {
    VT a = (VT)a_, b = (VT)b_, r;
    for (int i = 0; i < N; ++i) r[i] = a[i] > b[i] ? a[i] - b[i] : 0;
    return (V128)r;
}
OP V128 op_VectorUnsignedSaturatedSub16(V128 a, V128 b) { return SatSubU<u16x8, u16, 8>(a, b); }
OP V128 op_VectorSignedSaturatedAdd16(V128 a_, V128 b_) {
    s16x8 a = (s16x8)a_, b = (s16x8)b_, r;
    for (int i = 0; i < 8; ++i) { int s = a[i] + b[i]; r[i] = (s16)(s > 32767 ? 32767 : s < -32768 ? -32768 : s); }
    return (V128)r;
}
OP V128 op_VectorSignedAbsoluteDifference16(V128 a_, V128 b_) {
    s16x8 a = (s16x8)a_, b = (s16x8)b_, r;
    for (int i = 0; i < 8; ++i) { int d = a[i] - b[i]; r[i] = (s16)(d < 0 ? -d : d); }
    return (V128)r;
}
OP V128 op_VectorCountLeadingZeros32(V128 a_) {
    u32x4 a = (u32x4)a_, r;
    for (int i = 0; i < 4; ++i) r[i] = a[i] ? __builtin_clz(a[i]) : 32;
    return (V128)r;
}
OP V128 op_VectorReduceAdd32(V128 a_) { u32x4 a = (u32x4)a_; u32x4 r = {a[0] + a[1] + a[2] + a[3], 0, 0, 0}; return (V128)r; }
OP V128 op_VectorPairedAdd32(V128 a_, V128 b_) {
    u32x4 a = (u32x4)a_, b = (u32x4)b_;
    u32x4 r = {a[0] + a[1], a[2] + a[3], b[0] + b[1], b[2] + b[3]};
    return (V128)r;
}
OP V128 op_VectorPairedAddLower32(V128 a_, V128 b_) {
    u32x4 a = (u32x4)a_, b = (u32x4)b_;
    u32x4 r = {a[0] + a[1], b[0] + b[1], 0, 0};
    return (V128)r;
}
OP V128 op_VectorPairedMinLowerS32(V128 a_, V128 b_) {
    s32x4 a = (s32x4)a_, b = (s32x4)b_;
    s32x4 r = {a[0] < a[1] ? a[0] : a[1], b[0] < b[1] ? b[0] : b[1], 0, 0};
    return (V128)r;
}
OP V128 op_VectorPairedMaxLowerS32(V128 a_, V128 b_) {
    s32x4 a = (s32x4)a_, b = (s32x4)b_;
    s32x4 r = {a[0] > a[1] ? a[0] : a[1], b[0] > b[1] ? b[0] : b[1], 0, 0};
    return (V128)r;
}
OP V128 op_VectorPairedAddUnsignedWiden16(V128 a_) {
    u16x8 a = (u16x8)a_; u32x4 r;
    for (int i = 0; i < 4; ++i) r[i] = (u32)a[2 * i] + a[2 * i + 1];
    return (V128)r;
}
OP V128 op_VectorMultiplyUnsignedWiden16(V128 a_, V128 b_) {
    u16x8 a = (u16x8)a_, b = (u16x8)b_; u32x4 r;
    for (int i = 0; i < 4; ++i) r[i] = (u32)a[i] * b[i];
    return (V128)r;
}
OP V128 op_VectorMultiplySignedWiden16(V128 a_, V128 b_) {
    s16x8 a = (s16x8)a_, b = (s16x8)b_; s32x4 r;
    for (int i = 0; i < 4; ++i) r[i] = (s32)a[i] * b[i];
    return (V128)r;
}
OP V128 op_VectorMultiplyUnsignedWiden32(V128 a_, V128 b_) {
    u32x4 a = (u32x4)a_, b = (u32x4)b_;
    return Q((u64)a[0] * b[0], (u64)a[1] * b[1]);
}

// ---- vectors: shifts -------------------------------------------------------------------------
template <typename VT, typename T, int N, int W> OP V128 ShlImm(V128 a_, u8 s) {
    VT a = (VT)a_, r; for (int i = 0; i < N; ++i) r[i] = s >= W ? 0 : (T)(a[i] << s); return (V128)r;
}
template <typename VT, typename T, int N, int W> OP V128 ShrImm(V128 a_, u8 s) {
    VT a = (VT)a_, r; for (int i = 0; i < N; ++i) r[i] = s >= W ? 0 : (T)(a[i] >> s); return (V128)r;
}
template <typename VT, typename T, int N, int W> OP V128 SarImm(V128 a_, u8 s) {
    VT a = (VT)a_, r; for (int i = 0; i < N; ++i) r[i] = (T)(a[i] >> (s >= W ? W - 1 : s)); return (V128)r;
}
OP V128 op_VectorLogicalShiftLeft8(V128 a, u8 s) { return ShlImm<u8x16, u8, 16, 8>(a, s); }
OP V128 op_VectorLogicalShiftLeft16(V128 a, u8 s) { return ShlImm<u16x8, u16, 8, 16>(a, s); }
OP V128 op_VectorLogicalShiftLeft32(V128 a, u8 s) { return ShlImm<u32x4, u32, 4, 32>(a, s); }
OP V128 op_VectorLogicalShiftLeft64(V128 a, u8 s) { return ShlImm<u64x2, u64, 2, 64>(a, s); }
OP V128 op_VectorLogicalShiftRight8(V128 a, u8 s) { return ShrImm<u8x16, u8, 16, 8>(a, s); }
OP V128 op_VectorLogicalShiftRight16(V128 a, u8 s) { return ShrImm<u16x8, u16, 8, 16>(a, s); }
OP V128 op_VectorLogicalShiftRight32(V128 a, u8 s) { return ShrImm<u32x4, u32, 4, 32>(a, s); }
OP V128 op_VectorLogicalShiftRight64(V128 a, u8 s) { return ShrImm<u64x2, u64, 2, 64>(a, s); }
OP V128 op_VectorArithmeticShiftRight8(V128 a, u8 s) { return SarImm<s8x16, s8, 16, 8>(a, s); }
OP V128 op_VectorArithmeticShiftRight16(V128 a, u8 s) { return SarImm<s16x8, s16, 8, 16>(a, s); }
OP V128 op_VectorArithmeticShiftRight32(V128 a, u8 s) { return SarImm<s32x4, s32, 4, 32>(a, s); }
OP V128 op_VectorArithmeticShiftRight64(V128 a, u8 s) { return SarImm<s64x2, s64, 2, 64>(a, s); }
// USHL/SSHL: the shift is the signed low byte of each element of b (negative shifts right).
template <typename UT, typename ST, typename T, typename S, int N, int W> OP V128 VShift(V128 a_, V128 b_, bool arith) {
    UT ua = (UT)a_; ST sa = (ST)a_; UT b = (UT)b_; UT r;
    for (int i = 0; i < N; ++i) {
        const int s = (s8)(u8)b[i];
        if (s >= 0) r[i] = s >= W ? 0 : (T)(ua[i] << s);
        else if (arith) r[i] = (T)(S)(sa[i] >> (-s >= W ? W - 1 : -s));
        else r[i] = -s >= W ? 0 : (T)(ua[i] >> -s);
    }
    return (V128)r;
}
OP V128 op_VectorLogicalVShift8(V128 a, V128 b) { return VShift<u8x16, s8x16, u8, s8, 16, 8>(a, b, false); }
OP V128 op_VectorLogicalVShift16(V128 a, V128 b) { return VShift<u16x8, s16x8, u16, s16, 8, 16>(a, b, false); }
OP V128 op_VectorLogicalVShift32(V128 a, V128 b) { return VShift<u32x4, s32x4, u32, s32, 4, 32>(a, b, false); }
OP V128 op_VectorLogicalVShift64(V128 a, V128 b) { return VShift<u64x2, s64x2, u64, s64, 2, 64>(a, b, false); }
OP V128 op_VectorArithmeticVShift8(V128 a, V128 b) { return VShift<u8x16, s8x16, u8, s8, 16, 8>(a, b, true); }
OP V128 op_VectorArithmeticVShift16(V128 a, V128 b) { return VShift<u16x8, s16x8, u16, s16, 8, 16>(a, b, true); }
OP V128 op_VectorArithmeticVShift32(V128 a, V128 b) { return VShift<u32x4, s32x4, u32, s32, 4, 32>(a, b, true); }
OP V128 op_VectorArithmeticVShift64(V128 a, V128 b) { return VShift<u64x2, s64x2, u64, s64, 2, 64>(a, b, true); }

// ---- vectors: permutes -----------------------------------------------------------------------
OP u128 U(V128 v) { return (u128)Lo(v) | ((u128)Hi(v) << 64); }
OP V128 FromU(u128 v) { return Q((u64)v, (u64)(v >> 64)); }
OP V128 op_VectorExtract(V128 a, V128 b, u8 pos) {
    if (pos == 0) return a;
    return FromU((U(a) >> pos) | (U(b) << (128 - pos)));
}
OP V128 op_VectorExtractLower(V128 a, V128 b, u8 pos) {
    const u128 joined = (u128)Lo(a) | ((u128)Lo(b) << 64);
    return Q((u64)(joined >> pos), 0);
}
OP V128 op_VectorRotateWholeVectorRight(V128 a, u8 amount) {
    amount &= 127;
    if (!amount) return a;
    return FromU((U(a) >> amount) | (U(a) << (128 - amount)));
}
template <typename VT, int N> OP V128 ZipLo(V128 a_, V128 b_) {
    VT a = (VT)a_, b = (VT)b_, r; for (int i = 0; i < N / 2; ++i) { r[2 * i] = a[i]; r[2 * i + 1] = b[i]; } return (V128)r;
}
template <typename VT, int N> OP V128 ZipHi(V128 a_, V128 b_) {
    VT a = (VT)a_, b = (VT)b_, r; for (int i = 0; i < N / 2; ++i) { r[2 * i] = a[N / 2 + i]; r[2 * i + 1] = b[N / 2 + i]; } return (V128)r;
}
OP V128 op_VectorInterleaveLower8(V128 a, V128 b) { return ZipLo<u8x16, 16>(a, b); }
OP V128 op_VectorInterleaveLower16(V128 a, V128 b) { return ZipLo<u16x8, 8>(a, b); }
OP V128 op_VectorInterleaveLower32(V128 a, V128 b) { return ZipLo<u32x4, 4>(a, b); }
OP V128 op_VectorInterleaveLower64(V128 a, V128 b) { return ZipLo<u64x2, 2>(a, b); }
OP V128 op_VectorInterleaveUpper8(V128 a, V128 b) { return ZipHi<u8x16, 16>(a, b); }
OP V128 op_VectorInterleaveUpper16(V128 a, V128 b) { return ZipHi<u16x8, 8>(a, b); }
OP V128 op_VectorInterleaveUpper32(V128 a, V128 b) { return ZipHi<u32x4, 4>(a, b); }
OP V128 op_VectorInterleaveUpper64(V128 a, V128 b) { return ZipHi<u64x2, 2>(a, b); }
// UZP1/UZP2 over 128-bit inputs, and the 64-bit (Lower) forms with a zero upper half.
template <typename VT, int N> OP V128 Uzp(V128 a_, V128 b_, int odd) {
    VT a = (VT)a_, b = (VT)b_, r; for (int i = 0; i < N / 2; ++i) { r[i] = a[2 * i + odd]; r[N / 2 + i] = b[2 * i + odd]; } return (V128)r;
}
template <typename VT, int N> OP V128 UzpLower(V128 a_, V128 b_, int odd) {
    VT a = (VT)a_, b = (VT)b_, r{}; for (int i = 0; i < N / 4; ++i) { r[i] = a[2 * i + odd]; r[N / 4 + i] = b[2 * i + odd]; } return (V128)r;
}
OP V128 op_VectorDeinterleaveEven8(V128 a, V128 b) { return Uzp<u8x16, 16>(a, b, 0); }
OP V128 op_VectorDeinterleaveEven16(V128 a, V128 b) { return Uzp<u16x8, 8>(a, b, 0); }
OP V128 op_VectorDeinterleaveEven32(V128 a, V128 b) { return Uzp<u32x4, 4>(a, b, 0); }
OP V128 op_VectorDeinterleaveOdd8(V128 a, V128 b) { return Uzp<u8x16, 16>(a, b, 1); }
OP V128 op_VectorDeinterleaveOdd16(V128 a, V128 b) { return Uzp<u16x8, 8>(a, b, 1); }
OP V128 op_VectorDeinterleaveOdd32(V128 a, V128 b) { return Uzp<u32x4, 4>(a, b, 1); }
OP V128 op_VectorDeinterleaveEvenLower8(V128 a, V128 b) { return UzpLower<u8x16, 16>(a, b, 0); }
OP V128 op_VectorDeinterleaveEvenLower16(V128 a, V128 b) { return UzpLower<u16x8, 8>(a, b, 0); }
OP V128 op_VectorDeinterleaveEvenLower32(V128 a, V128 b) { return UzpLower<u32x4, 4>(a, b, 0); }
OP V128 op_VectorDeinterleaveOddLower8(V128 a, V128 b) { return UzpLower<u8x16, 16>(a, b, 1); }
OP V128 op_VectorDeinterleaveOddLower16(V128 a, V128 b) { return UzpLower<u16x8, 8>(a, b, 1); }
OP V128 op_VectorDeinterleaveOddLower32(V128 a, V128 b) { return UzpLower<u32x4, 4>(a, b, 1); }
OP V128 op_VectorTranspose32(V128 a_, V128 b_, bool part) {
    u32x4 a = (u32x4)a_, b = (u32x4)b_; const int p = part;
    u32x4 r = {a[p], b[p], a[2 + p], b[2 + p]};
    return (V128)r;
}
template <typename VT, int N, int G> OP V128 Rev64(V128 a_) {
    VT a = (VT)a_, r; for (int i = 0; i < N; ++i) r[i] = a[(i / G) * G + (G - 1 - i % G)]; return (V128)r;
}
OP V128 op_VectorReverseElementsInLongGroups8(V128 a) { return Rev64<u8x16, 16, 8>(a); }
OP V128 op_VectorReverseElementsInLongGroups16(V128 a) { return Rev64<u16x8, 8, 4>(a); }
OP V128 op_VectorReverseElementsInLongGroups32(V128 a) { return Rev64<u32x4, 4, 2>(a); }

// ---- vectors: widen / narrow -----------------------------------------------------------------
OP V128 op_VectorZeroExtend8(V128 a_) { u8x16 a = (u8x16)a_; u16x8 r; for (int i = 0; i < 8; ++i) r[i] = a[i]; return (V128)r; }
OP V128 op_VectorZeroExtend16(V128 a_) { u16x8 a = (u16x8)a_; u32x4 r; for (int i = 0; i < 4; ++i) r[i] = a[i]; return (V128)r; }
OP V128 op_VectorZeroExtend32(V128 a_) { u32x4 a = (u32x4)a_; return Q(a[0], a[1]); }
OP V128 op_VectorZeroExtend64(V128 a) { return Q(Lo(a), 0); }
OP V128 op_VectorSignExtend8(V128 a_) { s8x16 a = (s8x16)a_; s16x8 r; for (int i = 0; i < 8; ++i) r[i] = a[i]; return (V128)r; }
OP V128 op_VectorSignExtend16(V128 a_) { s16x8 a = (s16x8)a_; s32x4 r; for (int i = 0; i < 4; ++i) r[i] = a[i]; return (V128)r; }
OP V128 op_VectorSignExtend32(V128 a_) { s32x4 a = (s32x4)a_; return Q((u64)(s64)a[0], (u64)(s64)a[1]); }
OP V128 op_VectorNarrow16(V128 a_) { u16x8 a = (u16x8)a_; u8x16 r{}; for (int i = 0; i < 8; ++i) r[i] = (u8)a[i]; return (V128)r; }
OP V128 op_VectorNarrow32(V128 a_) { u32x4 a = (u32x4)a_; u16x8 r{}; for (int i = 0; i < 4; ++i) r[i] = (u16)a[i]; return (V128)r; }
OP V128 op_VectorNarrow64(V128 a_) { u64x2 a = (u64x2)a_; u32x4 r = {(u32)a[0], (u32)a[1], 0, 0}; return (V128)r; }
template <typename T> OP T Clamp(long long v, long long lo, long long hi) { return (T)(v < lo ? lo : v > hi ? hi : v); }
OP V128 op_VectorSignedSaturatedNarrowToUnsigned16(V128 a_) { s16x8 a = (s16x8)a_; u8x16 r{}; for (int i = 0; i < 8; ++i) r[i] = Clamp<u8>(a[i], 0, 255); return (V128)r; }
OP V128 op_VectorSignedSaturatedNarrowToUnsigned32(V128 a_) { s32x4 a = (s32x4)a_; u16x8 r{}; for (int i = 0; i < 4; ++i) r[i] = Clamp<u16>(a[i], 0, 65535); return (V128)r; }
OP V128 op_VectorSignedSaturatedNarrowToSigned16(V128 a_) { s16x8 a = (s16x8)a_; s8x16 r{}; for (int i = 0; i < 8; ++i) r[i] = Clamp<s8>(a[i], -128, 127); return (V128)r; }
OP V128 op_VectorSignedSaturatedNarrowToSigned32(V128 a_) { s32x4 a = (s32x4)a_; s16x8 r{}; for (int i = 0; i < 4; ++i) r[i] = Clamp<s16>(a[i], -32768, 32767); return (V128)r; }
OP V128 op_VectorUnsignedSaturatedNarrow16(V128 a_) { u16x8 a = (u16x8)a_; u8x16 r{}; for (int i = 0; i < 8; ++i) r[i] = a[i] > 255 ? 255 : a[i]; return (V128)r; }
OP V128 op_VectorUnsignedSaturatedNarrow32(V128 a_) { u32x4 a = (u32x4)a_; u16x8 r{}; for (int i = 0; i < 4; ++i) r[i] = a[i] > 65535 ? 65535 : a[i]; return (V128)r; }

// ---- vectors: table lookup -------------------------------------------------------------------
struct NoVec {};
struct VTable { V128 t[4]; int n = 0; };
OP void TableAdd(VTable& t, V128 v) { t.t[t.n++] = v; }
OP void TableAdd(VTable&, NoVec) {}
template <typename A, typename B, typename C, typename D> OP VTable op_VectorTable(A a, B b, C c, D d) {
    VTable t; TableAdd(t, a); TableAdd(t, b); TableAdd(t, c); TableAdd(t, d); return t;
}
OP V128 op_VectorTableLookup128(V128 defaults, VTable t, V128 indices) {
    u8x16 r = (u8x16)defaults, idx = (u8x16)indices;
    for (int i = 0; i < 16; ++i) {
        const int j = idx[i];
        if (j < t.n * 16) r[i] = ((u8x16)t.t[j / 16])[j % 16];
    }
    return (V128)r;
}
OP V128 op_VectorTableLookup64(V128 defaults, VTable t, V128 indices) {
    u8x16 r = (u8x16)defaults, idx = (u8x16)indices;
    for (int i = 0; i < 8; ++i) {
        const int j = idx[i];
        if (j < t.n * 8) r[i] = ((u8x16)t.t[j / 8])[j % 8];
    }
    return Q(Lo((V128)r), 0);
}

// ---- vectors: floating point -----------------------------------------------------------------
#define FPLANE2(NAME, VT, N, EXPR) \
    OP V128 op_##NAME(V128 a_, V128 b_, bool) { VT a = (VT)a_, b = (VT)b_, r; for (int i = 0; i < N; ++i) r[i] = (EXPR); return (V128)r; }
FPLANE2(FPVectorAdd32, f32x4, 4, a[i] + b[i])
FPLANE2(FPVectorAdd64, f64x2, 2, a[i] + b[i])
FPLANE2(FPVectorSub32, f32x4, 4, a[i] - b[i])
FPLANE2(FPVectorSub64, f64x2, 2, a[i] - b[i])
FPLANE2(FPVectorMul32, f32x4, 4, a[i] * b[i])
FPLANE2(FPVectorMul64, f64x2, 2, a[i] * b[i])
FPLANE2(FPVectorDiv32, f32x4, 4, a[i] / b[i])
FPLANE2(FPVectorDiv64, f64x2, 2, a[i] / b[i])
FPLANE2(FPVectorMax32, f32x4, 4, FMax(a[i], b[i]))
FPLANE2(FPVectorMax64, f64x2, 2, FMax(a[i], b[i]))
FPLANE2(FPVectorMin32, f32x4, 4, FMin(a[i], b[i]))
FPLANE2(FPVectorMin64, f64x2, 2, FMin(a[i], b[i]))
FPLANE2(FPVectorMaxNumeric32, f32x4, 4, FMaxNum(a[i], b[i]))
FPLANE2(FPVectorMinNumeric32, f32x4, 4, FMinNum(a[i], b[i]))
#define FPCMP(NAME, FT, UT, N, EXPR) \
    OP V128 op_##NAME(V128 a_, V128 b_, bool) { FT a = (FT)a_, b = (FT)b_; UT r; for (int i = 0; i < N; ++i) r[i] = (EXPR) ? (decltype(+r[i]))~0ULL : 0; return (V128)r; }
FPCMP(FPVectorEqual32, f32x4, u32x4, 4, a[i] == b[i])
FPCMP(FPVectorEqual64, f64x2, u64x2, 2, a[i] == b[i])
FPCMP(FPVectorGreater32, f32x4, u32x4, 4, a[i] > b[i])
FPCMP(FPVectorGreater64, f64x2, u64x2, 2, a[i] > b[i])
FPCMP(FPVectorGreaterEqual32, f32x4, u32x4, 4, a[i] >= b[i])
FPCMP(FPVectorGreaterEqual64, f64x2, u64x2, 2, a[i] >= b[i])
OP V128 op_FPVectorMulAdd32(V128 a_, V128 b_, V128 c_, bool) {
    f32x4 a = (f32x4)a_, b = (f32x4)b_, c = (f32x4)c_, r;
    for (int i = 0; i < 4; ++i) r[i] = std::fma(b[i], c[i], a[i]);
    return (V128)r;
}
OP V128 op_FPVectorMulAdd64(V128 a_, V128 b_, V128 c_, bool) {
    f64x2 a = (f64x2)a_, b = (f64x2)b_, c = (f64x2)c_, r;
    for (int i = 0; i < 2; ++i) r[i] = std::fma(b[i], c[i], a[i]);
    return (V128)r;
}
OP V128 op_FPVectorPairedAdd32(V128 a_, V128 b_, bool) {
    f32x4 a = (f32x4)a_, b = (f32x4)b_;
    f32x4 r = {a[0] + a[1], a[2] + a[3], b[0] + b[1], b[2] + b[3]};
    return (V128)r;
}
OP V128 op_FPVectorPairedAddLower32(V128 a_, V128 b_, bool) {
    f32x4 a = (f32x4)a_, b = (f32x4)b_;
    f32x4 r = {a[0] + a[1], b[0] + b[1], 0, 0};
    return Q(Lo((V128)r), 0);
}
OP V128 op_FPVectorAbs32(V128 a) { u32x4 m = {0x7fffffffu, 0x7fffffffu, 0x7fffffffu, 0x7fffffffu}; return (V128)((u32x4)a & m); }
OP V128 op_FPVectorAbs64(V128 a) { u64x2 m = {0x7fffffffffffffffULL, 0x7fffffffffffffffULL}; return (V128)((u64x2)a & m); }
OP V128 op_FPVectorNeg32(V128 a) { u32x4 m = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u}; return (V128)((u32x4)a ^ m); }
OP V128 op_FPVectorNeg64(V128 a) { u64x2 m = {0x8000000000000000ULL, 0x8000000000000000ULL}; return (V128)((u64x2)a ^ m); }
OP V128 op_FPVectorSqrt32(V128 a_, bool) { f32x4 a = (f32x4)a_, r; for (int i = 0; i < 4; ++i) r[i] = std::sqrt(a[i]); return (V128)r; }
OP V128 op_FPVectorSqrt64(V128 a_, bool) { f64x2 a = (f64x2)a_, r; for (int i = 0; i < 2; ++i) r[i] = std::sqrt(a[i]); return (V128)r; }
OP V128 op_FPVectorRecipEstimate32(V128 a_, bool) { f32x4 a = (f32x4)a_, r; for (int i = 0; i < 4; ++i) r[i] = 1.0f / a[i]; return (V128)r; }
OP V128 op_FPVectorRSqrtEstimate32(V128 a_, bool) { f32x4 a = (f32x4)a_, r; for (int i = 0; i < 4; ++i) r[i] = 1.0f / std::sqrt(a[i]); return (V128)r; }
OP V128 op_FPVectorRecipStepFused32(V128 a_, V128 b_, bool) {
    f32x4 a = (f32x4)a_, b = (f32x4)b_, r;
    for (int i = 0; i < 4; ++i)
        r[i] = ((std::isinf(a[i]) && b[i] == 0) || (a[i] == 0 && std::isinf(b[i]))) ? 2.0f : std::fma(-a[i], b[i], 2.0f);
    return (V128)r;
}
OP V128 op_FPVectorRSqrtStepFused32(V128 a_, V128 b_, bool) {
    f32x4 a = (f32x4)a_, b = (f32x4)b_, r;
    for (int i = 0; i < 4; ++i)
        r[i] = ((std::isinf(a[i]) && b[i] == 0) || (a[i] == 0 && std::isinf(b[i]))) ? 1.5f : std::fma(-a[i], b[i], 3.0f) * 0.5f;
    return (V128)r;
}
OP V128 op_FPVectorRoundInt32(V128 a_, u8 mode, bool, bool) { f32x4 a = (f32x4)a_, r; for (int i = 0; i < 4; ++i) r[i] = RoundInt(a[i], mode); return (V128)r; }
OP V128 op_FPVectorRoundInt64(V128 a_, u8 mode, bool, bool) { f64x2 a = (f64x2)a_, r; for (int i = 0; i < 2; ++i) r[i] = RoundInt(a[i], mode); return (V128)r; }
OP V128 op_FPVectorFromSignedFixed32(V128 a_, u8 f, u8, bool) { s32x4 a = (s32x4)a_; f32x4 r; for (int i = 0; i < 4; ++i) r[i] = FromFixed<float>(a[i], f); return (V128)r; }
OP V128 op_FPVectorFromUnsignedFixed32(V128 a_, u8 f, u8, bool) { u32x4 a = (u32x4)a_; f32x4 r; for (int i = 0; i < 4; ++i) r[i] = FromFixed<float>(a[i], f); return (V128)r; }
OP V128 op_FPVectorFromSignedFixed64(V128 a_, u8 f, u8, bool) { s64x2 a = (s64x2)a_; f64x2 r; for (int i = 0; i < 2; ++i) r[i] = FromFixed<double>(a[i], f); return (V128)r; }
OP V128 op_FPVectorFromUnsignedFixed64(V128 a_, u8 f, u8, bool) { u64x2 a = (u64x2)a_; f64x2 r; for (int i = 0; i < 2; ++i) r[i] = FromFixed<double>(a[i], f); return (V128)r; }
OP V128 op_FPVectorToSignedFixed32(V128 a_, u8 f, u8 m, bool) { f32x4 a = (f32x4)a_; s32x4 r; for (int i = 0; i < 4; ++i) r[i] = ToFixed<s32>(a[i], f, m); return (V128)r; }
OP V128 op_FPVectorToUnsignedFixed32(V128 a_, u8 f, u8 m, bool) { f32x4 a = (f32x4)a_; u32x4 r; for (int i = 0; i < 4; ++i) r[i] = ToFixed<u32>(a[i], f, m); return (V128)r; }
OP V128 op_FPVectorToSignedFixed64(V128 a_, u8 f, u8 m, bool) { f64x2 a = (f64x2)a_; s64x2 r; for (int i = 0; i < 2; ++i) r[i] = ToFixed<s64>(a[i], f, m); return (V128)r; }
OP V128 op_FPVectorToUnsignedFixed64(V128 a_, u8 f, u8 m, bool) { f64x2 a = (f64x2)a_; u64x2 r; for (int i = 0; i < 2; ++i) r[i] = ToFixed<u64>(a[i], f, m); return (V128)r; }

// ---- operations used by the SDK module ---------------------------------------------------------
#include <immintrin.h>
#include <wmmintrin.h>

OP u32 op_MinUnsigned32_(u32 a, u32 b) { return a < b ? a : b; }
OP u32 op_SignedSaturatedAdd32(u32 a, u32 b) { s64 s = (s64)(s32)a + (s32)b; return (u32)Clamp<s32>(s, INT32_MIN, INT32_MAX); }
OP u32 op_SignedSaturatedSub32(u32 a, u32 b) { s64 s = (s64)(s32)a - (s32)b; return (u32)Clamp<s32>(s, INT32_MIN, INT32_MAX); }
OP u32 flags_SignedSaturatedAdd32(u32 a, u32 b) { s64 s = (s64)(s32)a + (s32)b; return (s != (s32)s) << 28; }
OP u32 flags_SignedSaturatedSub32(u32 a, u32 b) { s64 s = (s64)(s32)a - (s32)b; return (s != (s32)s) << 28; }
OP u64 op_FPRoundInt64_(u64 a, u8 m, bool e) { return op_FPRoundInt64(a, m, e); }
OP u32 op_FPHalfToSingle(u16 a, u8) { return B32(_cvtsh_ss(a)); }
OP u16 op_FPSingleToHalf(u32 a, u8) { return (u16)_cvtss_sh(F32(a), _MM_FROUND_TO_NEAREST_INT); }

// AES: AESE/AESD rounds without the key addition (done separately), MixColumns and its inverse.
OP V128 op_AESEncryptSingleRound(V128 a) { return (V128)_mm_aesenclast_si128((__m128i)a, _mm_setzero_si128()); }
OP V128 op_AESDecryptSingleRound(V128 a) { return (V128)_mm_aesdeclast_si128((__m128i)a, _mm_setzero_si128()); }
OP V128 op_AESInverseMixColumns(V128 a) { return (V128)_mm_aesimc_si128((__m128i)a); }
OP u8 XTime(u8 x) { return (u8)((x << 1) ^ ((x & 0x80) ? 0x1b : 0)); }
OP V128 op_AESMixColumns(V128 a_) {
    u8x16 a = (u8x16)a_, r;
    for (int c = 0; c < 4; ++c) {
        const u8 s0 = a[4 * c], s1 = a[4 * c + 1], s2 = a[4 * c + 2], s3 = a[4 * c + 3];
        r[4 * c] = XTime(s0) ^ (XTime(s1) ^ s1) ^ s2 ^ s3;
        r[4 * c + 1] = s0 ^ XTime(s1) ^ (XTime(s2) ^ s2) ^ s3;
        r[4 * c + 2] = s0 ^ s1 ^ XTime(s2) ^ (XTime(s3) ^ s3);
        r[4 * c + 3] = (XTime(s0) ^ s0) ^ s1 ^ s2 ^ XTime(s3);
    }
    return (V128)r;
}

// SHA256H/SHA256H2 (part1 selects H), SHA256SU0, SHA256SU1, per the Arm pseudocode.
OP u32 Ror32(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
OP V128 op_SHA256Hash(V128 x_, V128 y_, V128 w_, bool part1) {
    u32x4 x = (u32x4)x_, y = (u32x4)y_, w = (u32x4)w_;
    for (int e = 0; e < 4; ++e) {
        const u32 chs = ((y[1] ^ y[2]) & y[0]) ^ y[2];
        const u32 maj = (x[0] & x[1]) | ((x[0] | x[1]) & x[2]);
        const u32 t = y[3] + (Ror32(y[0], 6) ^ Ror32(y[0], 11) ^ Ror32(y[0], 25)) + chs + w[e];
        x[3] = t + x[3];
        y[3] = t + (Ror32(x[0], 2) ^ Ror32(x[0], 13) ^ Ror32(x[0], 22)) + maj;
        const u32x4 nx = {y[3], x[0], x[1], x[2]};
        const u32x4 ny = {x[3], y[0], y[1], y[2]};
        x = nx;
        y = ny;
    }
    return (V128)(part1 ? x : y);
}
OP V128 op_SHA256MessageSchedule0(V128 d_, V128 n_) {
    u32x4 d = (u32x4)d_, n = (u32x4)n_, r;
    const u32 t[4] = {d[1], d[2], d[3], n[0]};
    for (int e = 0; e < 4; ++e) r[e] = (Ror32(t[e], 7) ^ Ror32(t[e], 18) ^ (t[e] >> 3)) + d[e];
    return (V128)r;
}
OP V128 op_SHA256MessageSchedule1(V128 d_, V128 n_, V128 m_) {
    u32x4 d = (u32x4)d_, n = (u32x4)n_, m = (u32x4)m_, r;
    const u32 t0[4] = {n[1], n[2], n[3], m[0]};
    u32 t1[2] = {m[2], m[3]};
    for (int e = 0; e < 2; ++e) r[e] = (Ror32(t1[e], 17) ^ Ror32(t1[e], 19) ^ (t1[e] >> 10)) + d[e] + t0[e];
    t1[0] = r[0];
    t1[1] = r[1];
    for (int e = 2; e < 4; ++e) r[e] = (Ror32(t1[e - 2], 17) ^ Ror32(t1[e - 2], 19) ^ (t1[e - 2] >> 10)) + d[e] + t0[e];
    return (V128)r;
}
OP V128 op_VectorPolynomialMultiplyLong64(V128 a, V128 b) {
    return (V128)_mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0x00);
}

LANEWISE2(VectorSub8_, u8x16, a - b)
LANEWISE2(VectorMultiply8, u8x16, a * b)
LANEWISE2(VectorMaxS16, s16x8, a > b ? a : b)
LANEWISE2(VectorMinS16, s16x8, a < b ? a : b)
LANEWISE2(VectorMaxU16, u16x8, a > b ? a : b)
LANEWISE2(VectorMinU16, u16x8, a < b ? a : b)
LANEWISE2(VectorMaxU8, u8x16, a > b ? a : b)
#define LANELOOP2(NAME, VT, N, EXPR) \
    OP V128 op_##NAME(V128 a_, V128 b_) { VT a = (VT)a_, b = (VT)b_, r; for (int i = 0; i < N; ++i) r[i] = (EXPR); return (V128)r; }
LANELOOP2(VectorHalvingAddS16, s16x8, 8, (s16)(((s32)a[i] + b[i]) >> 1))
LANELOOP2(VectorHalvingAddS32, s32x4, 4, (s32)(((s64)a[i] + b[i]) >> 1))
LANELOOP2(VectorHalvingAddU8, u8x16, 16, (u8)(((u32)a[i] + b[i]) >> 1))
LANELOOP2(VectorHalvingSubS16, s16x8, 8, (s16)(((s32)a[i] - b[i]) >> 1))
LANELOOP2(VectorHalvingSubS32, s32x4, 4, (s32)(((s64)a[i] - b[i]) >> 1))
LANELOOP2(VectorUnsignedAbsoluteDifference8, u8x16, 16, (u8)(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]))
LANELOOP2(VectorUnsignedAbsoluteDifference16, u16x8, 8, (u16)(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]))
LANELOOP2(VectorSignedSaturatedAdd32, s32x4, 4, Clamp<s32>((long long)a[i] + b[i], INT32_MIN, INT32_MAX))
LANELOOP2(VectorSignedSaturatedSub32, s32x4, 4, Clamp<s32>((long long)a[i] - b[i], INT32_MIN, INT32_MAX))
OP s64 SatAdd64(s64 a, s64 b) { s64 r; return __builtin_add_overflow(a, b, &r) ? (a < 0 ? INT64_MIN : INT64_MAX) : r; }
LANELOOP2(VectorSignedSaturatedAdd64, s64x2, 2, SatAdd64(a[i], b[i]))
LANELOOP2(VectorSignedSaturatedDoublingMultiplyHigh16, s16x8, 8, Clamp<s16>(((long long)a[i] * b[i] * 2) >> 16, -32768, 32767))
LANELOOP2(VectorSignedSaturatedDoublingMultiplyHigh32, s32x4, 4, (s32)Clamp<s64>((long long)(((__int128)a[i] * b[i] * 2) >> 32), INT32_MIN, INT32_MAX))
LANELOOP2(VectorSignedSaturatedDoublingMultiplyHighRounding32, s32x4, 4, (s32)Clamp<s64>((long long)(((__int128)a[i] * b[i] * 2 + ((__int128)1 << 31)) >> 32), INT32_MIN, INT32_MAX))
OP V128 op_VectorSignedSaturatedDoublingMultiplyLong16(V128 a_, V128 b_) {
    s16x8 a = (s16x8)a_, b = (s16x8)b_; s32x4 r;
    for (int i = 0; i < 4; ++i) r[i] = Clamp<s32>((long long)a[i] * b[i] * 2, INT32_MIN, INT32_MAX);
    return (V128)r;
}
OP V128 op_VectorSignedSaturatedNarrowToSigned64(V128 a_) { s64x2 a = (s64x2)a_; s32x4 r{}; for (int i = 0; i < 2; ++i) r[i] = Clamp<s32>(a[i], INT32_MIN, INT32_MAX); return (V128)r; }
OP V128 op_VectorSignedSaturatedShiftLeftUnsigned16(V128 a_, u8 s) {
    s16x8 a = (s16x8)a_; u16x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] < 0 ? 0 : Clamp<u16>((long long)a[i] << s, 0, 65535);
    return (V128)r;
}
// SSHL/SQSHL/SRSHL/SQRSHL by the signed low byte of each element of b.
template <bool Round, bool Sat> OP s64 ShiftS(s64 v, int s, int width) {
    const s64 lo = -((s64)1 << (width - 1)), hi = ((s64)1 << (width - 1)) - 1;
    if (s >= 0) {
        if (s >= width) return Sat ? (v > 0 ? hi : v < 0 ? lo : 0) : 0;
        const __int128 r = (__int128)v << s;
        if (Sat) return r > hi ? hi : r < lo ? lo : (s64)r;
        return (s64)(((u64)v << s) & (width == 64 ? ~0ULL : (((u64)1 << width) - 1)));
    }
    s = -s;
    if (s >= width + 1) return Round ? 0 : (v < 0 ? -1 : 0);
    if (Round) return (s64)(((__int128)v + ((__int128)1 << (s - 1))) >> s);
    return v >> (s > 63 ? 63 : s);
}
#define VSHIFTS(NAME, VT, N, W, ROUND, SAT) \
    OP V128 op_##NAME(V128 a_, V128 b_) { VT a = (VT)a_, b = (VT)b_, r; \
        for (int i = 0; i < N; ++i) r[i] = (decltype(+r[i]))ShiftS<ROUND, SAT>(a[i], (s8)(u8)b[i], W); return (V128)r; }
VSHIFTS(VectorRoundingShiftLeftS32, s32x4, 4, 32, true, false)
VSHIFTS(VectorRoundingShiftLeftS64, s64x2, 2, 64, true, false)
VSHIFTS(VectorSignedSaturatedShiftLeft32, s32x4, 4, 32, false, true)
VSHIFTS(VectorSignedSaturatedRoundingShiftLeft32, s32x4, 4, 32, true, true)
OP V128 op_VectorAbs16(V128 a_) { s16x8 a = (s16x8)a_, r; for (int i = 0; i < 8; ++i) r[i] = (s16)(a[i] < 0 ? -a[i] : a[i]); return (V128)r; }
OP V128 op_VectorCountLeadingZeros16(V128 a_) { u16x8 a = (u16x8)a_, r; for (int i = 0; i < 8; ++i) r[i] = a[i] ? __builtin_clz(a[i]) - 16 : 16; return (V128)r; }
OP V128 op_VectorPopulationCount(V128 a_) { u8x16 a = (u8x16)a_, r; for (int i = 0; i < 16; ++i) r[i] = __builtin_popcount(a[i]); return (V128)r; }
OP V128 op_VectorReduceAdd8(V128 a_) { u8x16 a = (u8x16)a_; u8 s = 0; for (int i = 0; i < 16; ++i) s += a[i]; u8x16 r{}; r[0] = s; return (V128)r; }
OP V128 op_VectorReverseElementsInHalfGroups8(V128 a) { return Rev64<u8x16, 16, 2>(a); }
OP V128 op_VectorReverseElementsInWordGroups8(V128 a) { return Rev64<u8x16, 16, 4>(a); }
OP V128 op_VectorReverseElementsInWordGroups16(V128 a) { return Rev64<u16x8, 8, 2>(a); }
OP V128 op_VectorPairedAdd8(V128 a_, V128 b_) {
    u8x16 a = (u8x16)a_, b = (u8x16)b_, r;
    for (int i = 0; i < 8; ++i) { r[i] = a[2 * i] + a[2 * i + 1]; r[8 + i] = b[2 * i] + b[2 * i + 1]; }
    return (V128)r;
}
OP V128 op_VectorPairedAdd16(V128 a_, V128 b_) {
    u16x8 a = (u16x8)a_, b = (u16x8)b_, r;
    for (int i = 0; i < 4; ++i) { r[i] = a[2 * i] + a[2 * i + 1]; r[4 + i] = b[2 * i] + b[2 * i + 1]; }
    return (V128)r;
}
OP V128 op_VectorPairedAddUnsignedWiden8(V128 a_) { u8x16 a = (u8x16)a_; u16x8 r; for (int i = 0; i < 8; ++i) r[i] = (u16)a[2 * i] + a[2 * i + 1]; return (V128)r; }
OP V128 op_VectorPairedAddUnsignedWiden32(V128 a_) { u32x4 a = (u32x4)a_; return Q((u64)a[0] + a[1], (u64)a[2] + a[3]); }
OP V128 op_VectorPairedAddSignedWiden16(V128 a_) { s16x8 a = (s16x8)a_; s32x4 r; for (int i = 0; i < 4; ++i) r[i] = (s32)a[2 * i] + a[2 * i + 1]; return (V128)r; }
OP V128 op_VectorPairedAddSignedWiden32(V128 a_) { s32x4 a = (s32x4)a_; return Q((u64)((s64)a[0] + a[1]), (u64)((s64)a[2] + a[3])); }
OP V128 op_VectorPairedMaxLowerU16(V128 a_, V128 b_) {
    u16x8 a = (u16x8)a_, b = (u16x8)b_, r{};
    for (int i = 0; i < 2; ++i) { r[i] = a[2 * i] > a[2 * i + 1] ? a[2 * i] : a[2 * i + 1]; r[2 + i] = b[2 * i] > b[2 * i + 1] ? b[2 * i] : b[2 * i + 1]; }
    return (V128)r;
}
OP V128 op_VectorPairedMaxLowerU32(V128 a_, V128 b_) {
    u32x4 a = (u32x4)a_, b = (u32x4)b_;
    u32x4 r = {a[0] > a[1] ? a[0] : a[1], b[0] > b[1] ? b[0] : b[1], 0, 0};
    return (V128)r;
}
OP V128 op_VectorMultiplySignedWiden32(V128 a_, V128 b_) {
    s32x4 a = (s32x4)a_, b = (s32x4)b_;
    return Q((u64)((s64)a[0] * b[0]), (u64)((s64)a[1] * b[1]));
}
template <typename VT, int N> OP V128 Trn(V128 a_, V128 b_, bool part) {
    VT a = (VT)a_, b = (VT)b_, r; const int p = part;
    for (int i = 0; i < N / 2; ++i) { r[2 * i] = a[2 * i + p]; r[2 * i + 1] = b[2 * i + p]; }
    return (V128)r;
}
OP V128 op_VectorTranspose8(V128 a, V128 b, bool p) { return Trn<u8x16, 16>(a, b, p); }
OP V128 op_VectorTranspose16(V128 a, V128 b, bool p) { return Trn<u16x8, 8>(a, b, p); }
OP V128 op_VectorTranspose64(V128 a, V128 b, bool p) { return Trn<u64x2, 2>(a, b, p); }
OP V128 op_FPVectorFromSignedFixed64_(V128 a, u8 f, u8 m, bool c) { return op_FPVectorFromSignedFixed64(a, f, m, c); }
LANEWISE2(VectorMultiply64, u64x2, a * b)

struct SlotEntry {
    u64 address;
    GuestFn* slot;
};
