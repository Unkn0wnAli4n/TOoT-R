// Windows, with NVIDIA DLSS built in: NVIDIA's SDK library (nvsdk_ngx_s.lib) is made with Microsoft's compiler and
// expects a few pieces of Microsoft's C++ runtime. This build uses another compiler and runtime, so the pieces that
// library actually calls are given here. (CMakeLists.txt also gives the linker empty stand-ins for the three
// Microsoft runtime libraries the SDK library names.)
#if defined(_WIN32) && defined(LONGSHOT_NGX)
#include <windows.h>
#include <cstdint>
#include <cstdlib>

extern "C" {
int _fltused = 0x9875;
// The stack-check value Microsoft's compiler puts in functions with buffers, and its check.
std::uintptr_t __security_cookie = 0x00002B992DDFA232ull;
void __security_check_cookie(std::uintptr_t cookie) { if (cookie != __security_cookie) std::abort(); }
// The unwinder's handler for such functions: nothing to do during a search.
EXCEPTION_DISPOSITION __GSHandlerCheck(PEXCEPTION_RECORD, PVOID, PCONTEXT, PVOID) { return ExceptionContinueSearch; }
EXCEPTION_DISPOSITION __GSHandlerCheck_EH(PEXCEPTION_RECORD, PVOID, PCONTEXT, PVOID) { return ExceptionContinueSearch; }

// Function-local statics made safely on first use (Microsoft's scheme: the guard is 0 before, -1 during, and
// afterwards the number of the initialisation that finished).
static SRWLOCK init_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE init_done = CONDITION_VARIABLE_INIT;
static int init_epoch = INT_MIN;
__declspec(thread) int _Init_thread_epoch = INT_MIN;
void _Init_thread_header(int* guard) {
    AcquireSRWLockExclusive(&init_lock);
    if (*guard == 0) *guard = -1;
    else {
        while (*guard == -1) SleepConditionVariableSRW(&init_done, &init_lock, 100, 0);
        if (*guard == 0) *guard = -1; else _Init_thread_epoch = init_epoch;
    }
    ReleaseSRWLockExclusive(&init_lock);
}
void _Init_thread_footer(int* guard) {
    AcquireSRWLockExclusive(&init_lock);
    *guard = ++init_epoch; _Init_thread_epoch = init_epoch;
    ReleaseSRWLockExclusive(&init_lock);
    WakeAllConditionVariable(&init_done);
}
void _Init_thread_abort(int* guard) {
    AcquireSRWLockExclusive(&init_lock);
    *guard = 0;
    ReleaseSRWLockExclusive(&init_lock);
    WakeAllConditionVariable(&init_done);
}
// Microsoft's names for operator new and operator delete.
void* ngx_msvc_new(std::size_t size) __asm__("??2@YAPEAX_K@Z");
void* ngx_msvc_new(std::size_t size) { void* p = std::malloc(size ? size : 1); if (!p) std::abort(); return p; }
void ngx_msvc_delete(void* p) __asm__("??3@YAXPEAX@Z");
void ngx_msvc_delete(void* p) { std::free(p); }
void ngx_msvc_delete_sized(void* p, std::size_t) __asm__("??3@YAXPEAX_K@Z");
void ngx_msvc_delete_sized(void* p, std::size_t) { std::free(p); }
// What Microsoft's containers call when memory runs out or a length is impossible: there is nobody to catch it here.
[[noreturn]] void ngx_msvc_bad_alloc() __asm__("?_Xbad_alloc@std@@YAXXZ");
void ngx_msvc_bad_alloc() { std::abort(); }
[[noreturn]] void ngx_msvc_length_error(const char*) __asm__("?_Xlength_error@std@@YAXPEBD@Z");
void ngx_msvc_length_error(const char*) { std::abort(); }
[[noreturn]] void ngx_msvc_out_of_range(const char*) __asm__("?_Xout_of_range@std@@YAXPEBD@Z");
void ngx_msvc_out_of_range(const char*) { std::abort(); }
}
#endif
