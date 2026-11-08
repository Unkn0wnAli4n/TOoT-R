// tootr: the assertion semantics of the Eden common/assert.h that dynarmic was built with.
// ASSERT is soft (AssertFailSoftImpl), UNREACHABLE is fatal (AssertFatalImpl). recompiler/dynarmic_hooks.cpp
// defines both; the recompiler turns a fatal one into a trapping stub for that function.
#pragma once
#include "common/logging.h"

void AssertFailSoftImpl();
[[noreturn]] void AssertFatalImpl();

#define ASSERT_MSG(_a_, ...)                                                                       \
    ([&]() __attribute__((noinline)) {                                                             \
        if (!(_a_)) [[unlikely]] AssertFailSoftImpl();                                             \
    }())
#define ASSERT(_a_) ASSERT_MSG(_a_, "{}", #_a_)
#define UNREACHABLE_MSG(...) AssertFatalImpl()
#define UNREACHABLE() UNREACHABLE_MSG("")
#define DEBUG_ASSERT(_a_) do {} while (0)
#define DEBUG_ASSERT_MSG(_a_, ...) do {} while (0)
#define UNIMPLEMENTED() ASSERT(false && "Unimplemented!")
#define UNIMPLEMENTED_MSG(...) ASSERT_MSG(false, __VA_ARGS__)
#define UNIMPLEMENTED_IF(cond) ASSERT((!(cond)) && "Unimplemented!")
#define UNIMPLEMENTED_IF_MSG(cond, ...) ASSERT_MSG(!(cond), __VA_ARGS__)
#define ASSERT_OR_EXECUTE_MSG(_a_, _b_, ...) do { ASSERT_MSG(_a_, __VA_ARGS__); if (!(_a_)) { _b_ } } while (0)
#define ASSERT_OR_EXECUTE(_a_, _b_) ASSERT_OR_EXECUTE_MSG(_a_, _b_, "{}", #_a_)
