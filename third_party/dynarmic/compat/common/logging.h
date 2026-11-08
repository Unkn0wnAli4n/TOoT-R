// tootr: dynarmic's frontend only logs from failed assertions; the offline recompiler drops them.
#pragma once
#include <fmt/format.h>
#define LOG_TRACE(cls, ...) ((void)0)
#define LOG_DEBUG(cls, ...) ((void)0)
#define LOG_INFO(cls, ...) ((void)0)
#define LOG_WARNING(cls, ...) ((void)0)
#define LOG_ERROR(cls, ...) ((void)0)
#define LOG_CRITICAL(cls, ...) ((void)0)
