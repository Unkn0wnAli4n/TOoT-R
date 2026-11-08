// Guest aborts (nn::diag): report the condition, location and result code before stopping, so a
// failed SDK or game check names itself instead of ending in an anonymous Break SVC.

#include "../native.h"

using namespace rt;

namespace {

const char* Str(u64 p) { return p ? reinterpret_cast<const char*>(p) : "?"; }

[[noreturn]] void Report(Cpu* c, const char* condition, const char* function, const char* file, int line, u64 result_ptr) {
    const u32 result = result_ptr ? *reinterpret_cast<const u32*>(result_ptr) : 0;
    Fatal("guest abort: %s in %s (%s:%d), result %#x (module %u, description %u), called from %s", condition,
          function, file, line, result, result & 0x1ff, (result >> 9) & 0x1fff, DescribeAddress(c->x[30]).c_str());
}

} // namespace

NATIVE("_ZN2nn4diag6detail9AbortImplEv") { Report(c, "?", "?", "?", 0, 0); }
NATIVE("_ZN2nn4diag6detail9AbortImplEPKNS_6ResultE") { Report(c, "result check", "?", "?", 0, c->x[0]); }
NATIVE("_ZN2nn4diag6detail9AbortImplEPKNS_6ResultEPKNS_2os17UserExceptionInfoE") { Report(c, "result check", "?", "?", 0, c->x[0]); }
NATIVE("_ZN2nn4diag6detail9AbortImplEPKcS3_S3_i") { Report(c, Str(c->x[0]), Str(c->x[1]), Str(c->x[2]), (int)c->x[3], 0); }
NATIVE("_ZN2nn4diag6detail9AbortImplEPKcS3_S3_iS3_z") { Report(c, Str(c->x[0]), Str(c->x[1]), Str(c->x[2]), (int)c->x[3], 0); }
NATIVE("_ZN2nn4diag6detail9AbortImplEPKcS3_S3_iPKNS_6ResultES3_z") { Report(c, Str(c->x[0]), Str(c->x[1]), Str(c->x[2]), (int)c->x[3], c->x[4]); }
NATIVE("_ZN2nn4diag6detail9AbortImplEPKcS3_S3_iPKNS_6ResultEPKNS_2os17UserExceptionInfoES3_z") { Report(c, Str(c->x[0]), Str(c->x[1]), Str(c->x[2]), (int)c->x[3], c->x[4]); }
NATIVE("_ZN2nn4diag6detail10VAbortImplEPKcS3_S3_iPKNS_6ResultEPKNS_2os17UserExceptionInfoES3_RSt9__va_list") { Report(c, Str(c->x[0]), Str(c->x[1]), Str(c->x[2]), (int)c->x[3], c->x[4]); }
