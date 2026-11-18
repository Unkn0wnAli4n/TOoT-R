#include "logfile.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#define dup _dup
#define dup2 _dup2
#define fileno _fileno
#define isatty _isatty
#else
#include <unistd.h>
#endif

#include <SDL3/SDL.h>

#include "platform.h"

namespace rt {
namespace {
std::string log_path;
int console_fd = -1; // (what stderr was before the file took its place)
}

const std::string& LogFilePath() { return log_path; }

// stderr (and stdout) are pointed at the file itself, so the last lines of a crash are in it: the handlers write
// with the process already dying. What goes into the file is passed on to the console the game was started from by
// a thread that follows the file (a terminal, the launcher's window, a development script's pipe).
void OpenLogFile() {
    if (std::getenv("OOT_NO_LOG_FILE")) return;
    namespace fs = std::filesystem;
    std::error_code ec;
    const char* named = std::getenv("OOT_LOG_FILE");
    const fs::path path = named && *named ? fs::path(named) : fs::path(DataDir()) / "logs" / "tootr.log";
    fs::create_directories(path.parent_path(), ec);
    if (!named) { // (the runs before this one)
        const fs::path dir = path.parent_path();
        fs::remove(dir / "tootr.3.log", ec);
        fs::rename(dir / "tootr.2.log", dir / "tootr.3.log", ec);
        fs::rename(dir / "tootr.1.log", dir / "tootr.2.log", ec);
        fs::rename(path, dir / "tootr.1.log", ec);
    }
    std::fflush(stdout); std::fflush(stderr);
    const int console = dup(fileno(stderr)); // (may be -1: a start without any console)
    FILE* file = std::fopen(path.string().c_str(), "ab");
    if (!file) { if (console >= 0) { const char* m = "[rt] the log file cannot be written\n"; (void)!write(console, m, (unsigned)std::strlen(m)); } return; }
    dup2(fileno(file), fileno(stderr));
    dup2(fileno(file), fileno(stdout));
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::setvbuf(stdout, nullptr, _IOLBF, 1 << 12);
    log_path = path.string(); console_fd = console;
    {
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char stamp[64]; std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        std::fprintf(stderr, "[rt] log of %s (%s)\n", stamp, log_path.c_str());
    }
    if (console >= 0) std::thread([console, path] {
        FILE* in = std::fopen(path.string().c_str(), "rb");
        if (!in) return;
        char buffer[4096];
        for (;;) {
            const size_t n = std::fread(buffer, 1, sizeof buffer, in);
            if (n) { size_t done = 0; while (done < n) { const auto w = write(console, buffer + done, (unsigned)(n - done)); if (w <= 0) return; done += (size_t)w; } }
            else { std::clearerr(in); std::this_thread::sleep_for(std::chrono::milliseconds(40)); }
        }
    }).detach();
}

void ShowFatal(const char* message) {
    // (the console too, at once: the thread that follows the file does not outlive the abort that comes next)
    if (console_fd >= 0) { const std::string line = std::string("[rt] FATAL: ") + message + "\n"; (void)!write(console_fd, line.data(), (unsigned)line.size()); }
    static std::atomic<bool> shown{false};
    if (shown.exchange(true) || std::getenv("OOT_NO_DIALOG") || log_path.empty()) return; // (no file kept: a development run that reads the console)
    std::string text = message;
    text += "\n\nThe log of this run is kept here; attach it when you report the problem:\n" + log_path;
#ifdef _WIN32
    MessageBoxA(nullptr, text.c_str(), "TOoT:R", MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
#else
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "TOoT:R", text.c_str(), nullptr);
#endif
}
} // namespace rt
