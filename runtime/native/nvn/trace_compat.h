// Diagnostic call-site tracing (backtrace, dladdr, thread names) on Linux; no-op stand-ins on Windows, where the
// traces using them (OOT_*_TRACE switches) report nothing instead of failing to build.
#pragma once
#ifdef _WIN32
#include <cstddef>
struct Dl_info { const char* dli_fname = nullptr; void* dli_fbase = nullptr; const char* dli_sname = nullptr; void* dli_saddr = nullptr; };
inline int dladdr(const void*, Dl_info*) { return 0; }
inline int backtrace(void**, int) { return 0; }
using pthread_t = unsigned long;
inline pthread_t pthread_self() { return 0; }
inline int pthread_getname_np(pthread_t, char*, size_t) { return 0; }
#else
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#endif
