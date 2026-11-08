#include <atomic>
// nn::time from the host clock. Layouts: PosixTime is s64 seconds; CalendarTime is
// {s16 year; s8 month, day, hour, minute, second; pad}; CalendarAdditionalInfo is
// {s32 day_of_week; s32 day_of_year; char tz[8]; bool dst; s32 utc_offset}; ClockSnapshot is 0xD0
// bytes with the user/network times at 0x40/0x48 and the steady clock point at 0x90.

#include <chrono>
#include <ctime>

#include "../native.h"

using namespace rt;

namespace {

s64 Now() { return (s64)std::time(nullptr); }

s64 SteadySeconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct CalendarTime {
    s16 year;
    s8 month, day, hour, minute, second;
    u8 pad;
};
static_assert(sizeof(CalendarTime) == 8);

CalendarTime ToCalendar(s64 t, bool utc, u8* info) {
    const time_t tt = (time_t)t;
    std::tm tm{};
#ifdef _WIN32
    if (utc) gmtime_s(&tm, &tt);
    else localtime_s(&tm, &tt);
    long bias = 0;
    _get_timezone(&bias); // seconds west of UTC
    const long gmtoff = utc ? 0 : -bias + (tm.tm_isdst > 0 ? 3600 : 0);
    const char* zone = "UTC";
#else
    if (utc) gmtime_r(&tt, &tm);
    else localtime_r(&tt, &tm);
    const long gmtoff = tm.tm_gmtoff;
    const char* zone = tm.tm_zone ? tm.tm_zone : "UTC";
#endif
    if (info) {
        std::memset(info, 0, 0x18);
        const s32 dow = tm.tm_wday, doy = tm.tm_yday;
        std::memcpy(info, &dow, 4);
        std::memcpy(info + 4, &doy, 4);
        std::strncpy(reinterpret_cast<char*>(info + 8), utc ? "UTC" : zone, 7);
        info[16] = tm.tm_isdst > 0;
        const s32 offset = (s32)gmtoff;
        std::memcpy(info + 20, &offset, 4);
    }
    return CalendarTime{(s16)(tm.tm_year + 1900), (s8)(tm.tm_mon + 1), (s8)tm.tm_mday, (s8)tm.tm_hour,
                        (s8)tm.tm_min, (s8)tm.tm_sec, 0};
}

s64 FromCalendarUtc(const CalendarTime& ct) {
    std::tm tm{};
    tm.tm_year = ct.year - 1900;
    tm.tm_mon = ct.month - 1;
    tm.tm_mday = ct.day;
    tm.tm_hour = ct.hour;
    tm.tm_min = ct.minute;
    tm.tm_sec = ct.second;
#ifdef _WIN32
    return (s64)_mkgmtime(&tm);
#else
    return (s64)timegm(&tm);
#endif
}

u64 Pack(const CalendarTime& ct) { u64 v; std::memcpy(&v, &ct, 8); return v; }

int Compare(const CalendarTime& a, const CalendarTime& b) {
    const s64 x = FromCalendarUtc(a), y = FromCalendarUtc(b);
    return x < y ? -1 : x > y ? 1 : 0;
}

} // namespace

NATIVE("_ZN2nn4time10InitializeEv") { Ret(c, 0); }
NATIVE("_ZN2nn4time8FinalizeEv") { Ret(c, 0); }
NATIVE("_ZN2nn4time13IsInitializedEv") { Ret(c, 1); }
NATIVE("_ZN2nn4time23StandardUserSystemClock14GetCurrentTimeEPNS0_9PosixTimeE") { *Ptr<s64>(c, 0) = Now(); Ret(c, 0); }
NATIVE("_ZN2nn4time26StandardNetworkSystemClock14GetCurrentTimeEPNS0_9PosixTimeE") { *Ptr<s64>(c, 0) = Now(); Ret(c, 0); }
NATIVE("_ZN2nn4time26StandardNetworkSystemClock3nowEv") { Ret(c, (u64)Now()); }
NATIVE("_ZN2nn4time19StandardSteadyClock3nowEv") {
    // time_point is a std::chrono-style seconds count returned in x0 (callers read x0 directly).
    Ret(c, (u64)SteadySeconds());
}
NATIVE("_ZN2nn4time14ToCalendarTimeEPNS0_12CalendarTimeEPNS0_22CalendarAdditionalInfoERKNS0_9PosixTimeE") {
    const CalendarTime ct = ToCalendar(*Ptr<s64>(c, 2), false, Ptr<u8>(c, 1));
    std::memcpy(Ptr(c, 0), &ct, 8);
    Ret(c, 0);
}
NATIVE("_ZN2nn4time19ToCalendarTimeInUtcERKNS0_9PosixTimeE") { Ret(c, Pack(ToCalendar(*Ptr<s64>(c, 0), true, nullptr))); }
NATIVE("_ZN2nn4time18ToPosixTimeFromUtcERKNS0_12CalendarTimeE") { Ret(c, (u64)FromCalendarUtc(*Ptr<CalendarTime>(c, 0))); }
NATIVE("_ZN2nn4time12CalendarTimemIERKNS_8TimeSpanE") {
    CalendarTime* ct = Ptr<CalendarTime>(c, 0);
    const s64 span_ns = *Ptr<s64>(c, 1);
    *ct = ToCalendar(FromCalendarUtc(*ct) - span_ns / 1000000000, true, nullptr);
    Ret(c, c->x[0]);
}
NATIVE("_ZNK2nn4time12CalendarTime7IsValidEv") {
    const CalendarTime& ct = *Ptr<CalendarTime>(c, 0);
    Ret(c, ct.month >= 1 && ct.month <= 12 && ct.day >= 1 && ct.day <= 31 && ct.hour >= 0 && ct.hour < 24 &&
               ct.minute >= 0 && ct.minute < 60 && ct.second >= 0 && ct.second < 60);
}
NATIVE("_ZN2nn4timegtERKNS0_12CalendarTimeES3_") { Ret(c, Compare(*Ptr<CalendarTime>(c, 0), *Ptr<CalendarTime>(c, 1)) > 0); }
NATIVE("_ZN2nn4timeltERKNS0_12CalendarTimeES3_") { Ret(c, Compare(*Ptr<CalendarTime>(c, 0), *Ptr<CalendarTime>(c, 1)) < 0); }

NATIVE("_ZN2nn4time13ClockSnapshotC1Ev") { std::memset(Ptr(c, 0), 0, 0xD0); }
NATIVE("_ZN2nn4time13ClockSnapshot29CreateWithStandardSystemClockEPS1_") {
    u8* s = Ptr<u8>(c, 0);
    std::memset(s, 0, 0xD0);
    const s64 now = Now(), steady = SteadySeconds();
    std::memcpy(s + 0x40, &now, 8);
    std::memcpy(s + 0x48, &now, 8);
    const CalendarTime ct = ToCalendar(now, false, s + 0x60);
    std::memcpy(s + 0x50, &ct, 8);
    std::memcpy(s + 0x58, &ct, 8);
    ToCalendar(now, false, s + 0x78);
    std::memcpy(s + 0x90, &steady, 8);
    std::memset(s + 0x98, 0x5a, 16);
    Ret(c, 0);
}
NATIVE("_ZN2nn4time20CalculateSpanBetweenEPNS_8TimeSpanERKNS0_13ClockSnapshotES5_") {
    s64 a, b;
    std::memcpy(&a, Ptr<u8>(c, 1) + 0x90, 8);
    std::memcpy(&b, Ptr<u8>(c, 2) + 0x90, 8);
    *Ptr<s64>(c, 0) = (b - a) * 1000000000LL;
    Ret(c, 0);
}
NATIVE("_ZN2nn4time48CalculateStandardUserSystemClockDifferenceByUserERKNS0_13ClockSnapshotES3_") {
    s64 a, b;
    std::memcpy(&a, Ptr<u8>(c, 0) + 0x40, 8);
    std::memcpy(&b, Ptr<u8>(c, 1) + 0x40, 8);
    Ret(c, (u64)((b - a) * 1000000000LL));
}

// nn::os::GetSystemTick, natively (the counter as rt_cntpct). The engine reads it at the end of
// each frame from main+0x346c76c with its graphics module in x19 (found by Hookshot); the module's
// nominal frame rate drives the real-time step (nvn/present.cpp).
u64 rt_cntpct();
namespace nvn { extern std::atomic<u64> graphics_module; }
NATIVE("_ZN2nn2os13GetSystemTickEv") {
    if (c->x[30] == 0x80346c770ULL) nvn::graphics_module.store(c->x[19], std::memory_order_relaxed);
    Ret(c, rt_cntpct());
}
