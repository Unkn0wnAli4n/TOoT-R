// Assertion hooks for the vendored dynarmic frontend (third_party/dynarmic/compat/common/assert.h).
#include <csetjmp>
#include <cstdlib>
// Decoding data as code (address-taken false positives) can trip dynarmic assertions; the
// recompiler turns the escape into a trapping stub for that one function.
void AssertFailSoftImpl() {}
thread_local std::jmp_buf* g_assert_escape = nullptr;
void AssertFatalImpl() {
    if (g_assert_escape) std::longjmp(*g_assert_escape, 1);
    std::abort();
}
