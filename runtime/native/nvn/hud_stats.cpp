// The stats HUD's hardware figures (hud_stats.h): GPU load, temperature, clock and power; CPU load per core, temperature
// and clock; memory; the battery. Sources: NVIDIA's management library (nvml, installed with the driver, loaded when
// present), amdgpu's files on Linux (the Steam Deck), hwmon and /proc on Linux, the system's counters on Windows.
#include "hud_stats.h"
#include "vk.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <dlfcn.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace nvn::hud {
namespace {
std::mutex mutex; Stats latest; std::atomic<s64> asked_ms{0};
s64 NowMs() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- NVIDIA: nvml, loaded when there ----
struct Nvml {
    using Dev = void*;
    int (*init)() = nullptr; int (*count)(unsigned*) = nullptr; int (*by_index)(unsigned, Dev*) = nullptr;
    int (*uuid)(Dev, char*, unsigned) = nullptr; int (*util)(Dev, unsigned*) = nullptr; int (*temp)(Dev, int, unsigned*) = nullptr;
    int (*clock)(Dev, int, unsigned*) = nullptr; int (*power)(Dev, unsigned*) = nullptr; int (*fan)(Dev, unsigned*) = nullptr;
    Dev dev = nullptr;
    bool Open() {
#ifdef _WIN32
        HMODULE m = LoadLibraryA("nvml.dll");
        if (!m) { char path[MAX_PATH]; if (GetSystemDirectoryA(path, MAX_PATH)) { std::strcat(path, "\\nvml.dll"); m = LoadLibraryA(path); } }
        if (!m) return false;
        auto get = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(m, n)); };
#else
        void* m = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!m) return false;
        auto get = [&](const char* n) { return dlsym(m, n); };
#endif
        init = reinterpret_cast<int (*)()>(get("nvmlInit_v2")); count = reinterpret_cast<int (*)(unsigned*)>(get("nvmlDeviceGetCount_v2"));
        by_index = reinterpret_cast<int (*)(unsigned, Dev*)>(get("nvmlDeviceGetHandleByIndex_v2"));
        uuid = reinterpret_cast<int (*)(Dev, char*, unsigned)>(get("nvmlDeviceGetUUID"));
        util = reinterpret_cast<int (*)(Dev, unsigned*)>(get("nvmlDeviceGetUtilizationRates")); // (struct {gpu, memory}: the first is the GPU's)
        temp = reinterpret_cast<int (*)(Dev, int, unsigned*)>(get("nvmlDeviceGetTemperature"));
        clock = reinterpret_cast<int (*)(Dev, int, unsigned*)>(get("nvmlDeviceGetClockInfo"));
        power = reinterpret_cast<int (*)(Dev, unsigned*)>(get("nvmlDeviceGetPowerUsage"));
        fan = reinterpret_cast<int (*)(Dev, unsigned*)>(get("nvmlDeviceGetFanSpeed"));
        if (!init || !count || !by_index || init() != 0) return false;
        // (the device the game draws with: by its UUID, as Vulkan gives it)
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &id;
        vkGetPhysicalDeviceProperties2(Vk().physical, &p2);
        char want[64]; const u8* u = id.deviceUUID;
        std::snprintf(want, sizeof want, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                      u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
        unsigned n = 0; count(&n);
        for (unsigned i = 0; i < n; ++i) {
            Dev d = nullptr; if (by_index(i, &d) != 0) continue;
            char s[96] = {}; if (uuid && uuid(d, s, sizeof s) == 0 && !strcasecmp_(s, want)) { dev = d; break; }
            if (!dev) dev = d; // (the first, if none matches)
        }
        return dev != nullptr;
    }
    static int strcasecmp_(const char* a, const char* b) { for (; *a && *b; ++a, ++b) if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return 1; return *a != *b; }
    void Read(Stats& s) const {
        unsigned v[2] = {}; if (util && util(dev, v) == 0) s.gpu_load = (float)v[0];
        unsigned t = 0; if (temp && temp(dev, 0, &t) == 0) s.gpu_temp = (float)t;
        unsigned c = 0; if (clock && clock(dev, 0, &c) == 0) s.gpu_clock = (float)c;
        unsigned mw = 0; if (power && power(dev, &mw) == 0) s.gpu_power = mw / 1000.0f;
        unsigned f = 0; if (fan && fan(dev, &f) == 0) s.gpu_fan = (float)f;
        s.gpu_source = "nvml";
    }
};

#ifndef _WIN32
// ---- Linux files ----
bool ReadNumber(const std::string& path, double& out) {
    FILE* f = std::fopen(path.c_str(), "r"); if (!f) return false;
    const bool ok = std::fscanf(f, "%lf", &out) == 1; std::fclose(f); return ok;
}
std::string ReadLine(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r"); if (!f) return {};
    char b[128] = {}; if (!std::fgets(b, sizeof b, f)) b[0] = 0; std::fclose(f);
    std::string s = b; while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back(); return s;
}
// (amdgpu: the card of the game's device - by vendor, the first with a load file)
struct Amd {
    std::string device, hwmon;
    bool Find() {
        std::error_code ec;
        for (auto& e : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
            const std::string name = e.path().filename().string();
            if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) continue;
            const std::string d = e.path().string() + "/device";
            if (ReadLine(d + "/vendor") != "0x1002") continue;
            double busy; if (!ReadNumber(d + "/gpu_busy_percent", busy)) continue;
            device = d;
            for (auto& h : std::filesystem::directory_iterator(d + "/hwmon", ec)) { hwmon = h.path().string(); break; }
            return true;
        }
        return false;
    }
    void Read(Stats& s) const {
        double v;
        if (ReadNumber(device + "/gpu_busy_percent", v)) s.gpu_load = (float)v;
        if (!hwmon.empty()) {
            if (ReadNumber(hwmon + "/temp1_input", v)) s.gpu_temp = (float)(v / 1000);
            if (ReadNumber(hwmon + "/freq1_input", v)) s.gpu_clock = (float)(v / 1e6);
            if (ReadNumber(hwmon + "/power1_average", v) || ReadNumber(hwmon + "/power1_input", v)) s.gpu_power = (float)(v / 1e6);
        }
        if (s.gpu_clock < 0) { // (pp_dpm_sclk: the line marked '*' is the clock now)
            if (FILE* f = std::fopen((device + "/pp_dpm_sclk").c_str(), "r")) { char line[64]; while (std::fgets(line, sizeof line, f)) if (std::strchr(line, '*')) { int lv, mhz; if (std::sscanf(line, "%d: %dMhz", &lv, &mhz) == 2) s.gpu_clock = (float)mhz; } std::fclose(f); }
        }
        s.gpu_source = "amdgpu";
    }
};
// (the CPU's temperature: the hwmon of a known CPU sensor)
std::string CpuTempFile() {
    std::error_code ec;
    for (auto& h : std::filesystem::directory_iterator("/sys/class/hwmon", ec)) {
        const std::string name = ReadLine(h.path().string() + "/name");
        if (name == "k10temp" || name == "zenpower" || name == "coretemp" || name == "cpu_thermal" || name == "acpitz") {
            const std::string f = h.path().string() + "/temp1_input"; double v; if (ReadNumber(f, v)) return f;
        }
    }
    return {};
}
// (the battery: the first power supply of type Battery that powers the PC itself)
std::string BatteryDir() {
    std::error_code ec;
    for (auto& p : std::filesystem::directory_iterator("/sys/class/power_supply", ec))
        if (ReadLine(p.path().string() + "/type") == "Battery" && ReadLine(p.path().string() + "/scope") != "Device") return p.path().string(); // (not a controller's or a mouse's)
    return {};
}
#endif

void Sample(Stats& s) {
    static Nvml nvml; static int nvml_state = 0; // (0 not tried, 1 open, -1 none)
    if (!nvml_state) nvml_state = Vk().properties.vendorID == 0x10de && nvml.Open() ? 1 : -1;
    if (nvml_state > 0) nvml.Read(s);
    static auto at = std::chrono::steady_clock::now(); const auto now = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(now - at).count(); at = now;
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
#ifdef _WIN32
    auto ticks = [](const FILETIME& f) { return (double)(((u64)f.dwHighDateTime << 32) | f.dwLowDateTime) * 1e-7; };
    static double game0 = -1, idle0 = 0, busy0 = 0;
    FILETIME created, exited, kernel, user;
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        const double game = ticks(kernel) + ticks(user);
        if (game0 >= 0 && wall > 0) s.game_cpu = (float)std::clamp((game - game0) / (wall * cores) * 100.0, 0.0, 100.0);
        game0 = game;
    }
    FILETIME idle, sk, su;
    if (GetSystemTimes(&idle, &sk, &su)) {
        const double i = ticks(idle), busy = ticks(sk) + ticks(su) - i;
        if (busy0 > 0 && busy + i - busy0 - idle0 > 0) s.cpu_load = (float)std::clamp((busy - busy0) / (busy + i - busy0 - idle0) * 100.0, 0.0, 100.0);
        busy0 = busy; idle0 = i;
    }
    { // (each core's load: the system's processor performance figures, through ntdll)
        struct Perf { LARGE_INTEGER idle, kernel, user, dpc, interrupt; ULONG count; };
        using Query = LONG(WINAPI*)(int, void*, ULONG, ULONG*);
        static Query query = reinterpret_cast<Query>(GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQuerySystemInformation"));
        static std::vector<Perf> before;
        if (query) {
            std::vector<Perf> p(cores); ULONG got = 0;
            if (query(8 /*SystemProcessorPerformanceInformation*/, p.data(), (ULONG)(p.size() * sizeof(Perf)), &got) == 0) {
                p.resize(got / sizeof(Perf));
                if (before.size() == p.size()) for (size_t c = 0; c < p.size(); ++c) {
                    const double all = (double)(p[c].kernel.QuadPart + p[c].user.QuadPart - before[c].kernel.QuadPart - before[c].user.QuadPart); // (kernel includes idle)
                    const double idle_c = (double)(p[c].idle.QuadPart - before[c].idle.QuadPart);
                    s.core_loads.push_back(all > 0 ? (float)std::clamp((all - idle_c) / all * 100.0, 0.0, 100.0) : 0.0f);
                }
                before = p;
            }
        }
    }
    PROCESS_MEMORY_COUNTERS pm{}; pm.cb = sizeof pm;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof pm)) s.game_ram = (double)pm.WorkingSetSize;
    MEMORYSTATUSEX m{}; m.dwLength = sizeof m;
    if (GlobalMemoryStatusEx(&m)) { s.total_ram = (double)m.ullTotalPhys; s.used_ram = (double)(m.ullTotalPhys - m.ullAvailPhys); }
    SYSTEM_POWER_STATUS ps{};
    if (GetSystemPowerStatus(&ps) && ps.BatteryFlag != 128 && ps.BatteryFlag != 255 && ps.BatteryLifePercent <= 100) { s.battery = ps.BatteryLifePercent; s.charging = ps.ACLineStatus == 1; }
#else
    static double game0 = -1;
    rusage r{};
    if (getrusage(RUSAGE_SELF, &r) == 0) {
        const double game = r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6;
        if (game0 >= 0 && wall > 0) s.game_cpu = (float)std::clamp((game - game0) / (wall * cores) * 100.0, 0.0, 100.0);
        game0 = game;
    }
    static std::vector<std::pair<unsigned long long, unsigned long long>> before; // (busy, all) per line: the total, then each core
    if (FILE* f = std::fopen("/proc/stat", "r")) {
        std::vector<std::pair<unsigned long long, unsigned long long>> now_v; char line[256];
        while (std::fgets(line, sizeof line, f) && !std::strncmp(line, "cpu", 3)) {
            unsigned long long v[8] = {}; char name[16];
            if (std::sscanf(line, "%15s %llu %llu %llu %llu %llu %llu %llu %llu", name, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) < 9) continue;
            const unsigned long long all = v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]; now_v.push_back({all - v[3] - v[4], all});
        }
        std::fclose(f);
        if (before.size() == now_v.size())
            for (size_t i = 0; i < now_v.size(); ++i) {
                const double all = (double)(now_v[i].second - before[i].second), busy = (double)(now_v[i].first - before[i].first);
                const float load = all > 0 ? (float)std::clamp(busy / all * 100.0, 0.0, 100.0) : 0.0f;
                if (i == 0) s.cpu_load = load; else s.core_loads.push_back(load);
            }
        before = now_v;
    }
    { // (the clock: the average of the cores' current frequencies)
        double sum = 0; int n = 0, v;
        for (unsigned c = 0; c < cores; ++c) { double khz; if (ReadNumber("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpufreq/scaling_cur_freq", khz)) { sum += khz / 1000; ++n; } }
        (void)v; if (n) s.cpu_clock = (float)(sum / n);
    }
    static const std::string cpu_temp = CpuTempFile();
    if (double t; !cpu_temp.empty() && ReadNumber(cpu_temp, t)) s.cpu_temp = (float)(t / 1000);
    if (FILE* f = std::fopen("/proc/self/statm", "r")) { unsigned long long size = 0, resident = 0; if (std::fscanf(f, "%llu %llu", &size, &resident) == 2) s.game_ram = (double)resident * (double)sysconf(_SC_PAGESIZE); std::fclose(f); }
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char key[64]; unsigned long long kib; double total = 0, available = 0, swap_total = 0, swap_free = 0;
        while (std::fscanf(f, "%63s %llu kB\n", key, &kib) == 2) {
            if (!std::strcmp(key, "MemTotal:")) total = kib * 1024.0; else if (!std::strcmp(key, "MemAvailable:")) available = kib * 1024.0;
            else if (!std::strcmp(key, "SwapTotal:")) swap_total = kib * 1024.0; else if (!std::strcmp(key, "SwapFree:")) swap_free = kib * 1024.0;
        }
        std::fclose(f);
        s.total_ram = total; s.used_ram = total - available; s.swap_used = swap_total - swap_free;
    }
    if (nvml_state < 0) { static Amd amd; static int amd_state = 0; if (!amd_state) amd_state = amd.Find() ? 1 : -1; if (amd_state > 0) amd.Read(s); }
    static const std::string battery = BatteryDir();
    if (!battery.empty()) {
        double v;
        if (ReadNumber(battery + "/capacity", v)) s.battery = (float)v;
        s.charging = ReadLine(battery + "/status") == "Charging";
        if (ReadNumber(battery + "/power_now", v)) s.battery_watts = (float)(v / 1e6);
        else { double cur, volt; if (ReadNumber(battery + "/current_now", cur) && ReadNumber(battery + "/voltage_now", volt)) s.battery_watts = (float)(cur * volt / 1e12); }
    }
#endif
}

void Start() {
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            for (;;) {
                // (only while the HUD is on screen: it asks for the figures every frame)
                if (NowMs() - asked_ms.load(std::memory_order_relaxed) > 2000) { std::this_thread::sleep_for(std::chrono::milliseconds(250)); continue; }
                Stats s; Sample(s);
                { std::lock_guard lock{mutex}; latest = std::move(s); }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }).detach();
    });
}
} // namespace

Stats Current() { asked_ms.store(NowMs(), std::memory_order_relaxed); Start(); std::lock_guard lock{mutex}; return latest; }
} // namespace nvn::hud
