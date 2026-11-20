#include "logfile.h"
#include "rt.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>
#include <string>
#include <vector>

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
        // (Windows refuses to rename a file another program has open - the log being read in an editor or sent in a
        // browser - and the last run's log was then lost from the names while this run's went after it in the same
        // file: it is copied instead, and this run's starts empty)
        if (ec && fs::exists(path)) {
            std::error_code copied;
            fs::copy_file(path, dir / "tootr.1.log", fs::copy_options::overwrite_existing, copied);
#ifdef _WIN32
            if (!copied) if (FILE* t = _wfopen(path.wstring().c_str(), L"wb")) std::fclose(t);
#else
            if (!copied) if (FILE* t = std::fopen(path.string().c_str(), "wb")) std::fclose(t);
#endif
        }
    }
    std::fflush(stdout); std::fflush(stderr);
    const int console = dup(fileno(stderr)); // (may be -1: a start without any console)
#ifdef _WIN32
    // A program started without a console has no descriptor behind stderr at all (fileno gives -2), so there is
    // nothing to point at the file: the streams themselves are opened on it. (v0.3.0 left the file empty.)
    if (!_wfreopen(path.wstring().c_str(), L"ab", stderr)) return;
    _wfreopen(path.wstring().c_str(), L"ab", stdout);
#else
    FILE* file = std::fopen(path.string().c_str(), "ab");
    if (!file) { if (console >= 0) { const char* m = "[rt] the log file cannot be written\n"; (void)!write(console, m, (unsigned)std::strlen(m)); } return; }
    dup2(fileno(file), fileno(stderr));
    dup2(fileno(file), fileno(stdout));
#endif
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::setvbuf(stdout, nullptr, _IOLBF, 1 << 12);
    log_path = path.string(); console_fd = console;
    {
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char stamp[64]; std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        rt::LogLine("[rt] log of %s (%s)", stamp, log_path.c_str());
    }
    if (console >= 0) std::thread([console, path] {
#ifdef _WIN32
        FILE* in = _wfopen(path.wstring().c_str(), L"rb");
#else
        FILE* in = std::fopen(path.string().c_str(), "rb");
#endif
        if (!in) return;
        char buffer[4096];
        for (;;) {
            const size_t n = std::fread(buffer, 1, sizeof buffer, in);
            if (n) { size_t done = 0; while (done < n) { const auto w = write(console, buffer + done, (unsigned)(n - done)); if (w <= 0) return; done += (size_t)w; } }
            else { std::clearerr(in); std::this_thread::sleep_for(std::chrono::milliseconds(40)); }
        }
    }).detach();
}

// The settings window's Restart Game: the same program with the same arguments and environment.
void RestartProcess() {
    std::fflush(stdout); std::fflush(stderr);
#ifdef _WIN32
    std::wstring line = GetCommandLineW();
    STARTUPINFOW si{}; si.cb = sizeof si; PROCESS_INFORMATION pi{};
    if (CreateProcessW(nullptr, line.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) { CloseHandle(pi.hThread); CloseHandle(pi.hProcess); }
    std::_Exit(0);
#else
    // (the new game takes the console this one was started from, not this one's log file)
    if (console_fd >= 0) { dup2(console_fd, 2); dup2(console_fd, 1); }
    std::vector<std::string> words; { std::string all; if (FILE* f = std::fopen("/proc/self/cmdline", "rb")) { char b[4096]; size_t n; while ((n = std::fread(b, 1, sizeof b, f)) > 0) all.append(b, n); std::fclose(f); }
        for (size_t at = 0; at < all.size();) { words.emplace_back(all.c_str() + at); at += words.back().size() + 1; } }
    std::vector<char*> argv; for (auto& w : words) argv.push_back(w.data()); argv.push_back(nullptr);
    if (!words.empty()) execv("/proc/self/exe", argv.data());
    std::_Exit(0);
#endif
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

// ---- log lines without the user's account name ----
// (the home folder - C:\Users\<name> or /home/<name> - is written as ~ in every line: paths in the log named the player's
// account, and logs are posted on public issue trackers. Both slash directions on Windows.)
#include <cstdarg>
namespace {
std::vector<std::string> HomeSpellings() {
    std::vector<std::string> out;
    const char* home = std::getenv("USERPROFILE");
    if (!home || !*home) home = std::getenv("HOME");
    if (!home || std::strlen(home) < 4) return out;
    std::string h = home;
    while (!h.empty() && (h.back() == '/' || h.back() == '\\')) h.pop_back();
    out.push_back(h);
    std::string forward = h; for (char& ch : forward) if (ch == '\\') ch = '/';
    if (forward != h) out.push_back(forward);
    std::string back = h; for (char& ch : back) if (ch == '/') ch = '\\';
    if (back != h) out.push_back(back);
    return out;
}
}
void rt::LogLine(const char* format, ...) {
    static const std::vector<std::string> homes = HomeSpellings();
    char stack[1024];
    va_list args; va_start(args, format);
    va_list again; va_copy(again, args);
    const int n = std::vsnprintf(stack, sizeof stack, format, args);
    va_end(args);
    std::string line;
    if (n < 0) { va_end(again); return; }
    if ((size_t)n < sizeof stack) line.assign(stack, (size_t)n);
    else { line.resize((size_t)n + 1); std::vsnprintf(line.data(), line.size(), format, again); line.resize((size_t)n); }
    va_end(again);
    for (const std::string& h : homes)
        for (size_t at = line.find(h); at != std::string::npos; at = line.find(h, at + 1)) line.replace(at, h.size(), "~");
    line += '\n';
    std::fwrite(line.data(), 1, line.size(), stderr);
}
