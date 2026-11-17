// TOoT:R installer for Windows (TOoT-R-Installer.exe; built by tools/package/windows.sh).
//
// Three pages in one window: choose your own dump of the game (.nup) and the folder to install into; watch the
// installation step by step, with what is being worked on and how long is left; play. The work is the built-in
// install.ps1's (MSYS2 build tools, the build, shortcuts): this window reads its output - the "@step <key>" and
// "@progress <percent> <text>" lines of install.ps1, installer/launcher.py and tools/build_game.py, and the compiler's
// "[n/m] ..." lines - and keeps all of it under "Show details".
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shlobj.h>

#include <string>
#include <vector>

#include "setup_script.h" // kInstallScript, kInstallScriptSize (install.ps1)

namespace {

enum : int { IdNup = 101, IdNupBrowse, IdDir, IdDirBrowse, IdLog, IdInstall, IdClose, IdPlay, IdDetails, IdBack, IdFolder };
constexpr UINT WmOutput = WM_APP + 1, WmDone = WM_APP + 2;
constexpr COLORREF Bg = RGB(0x12, 0x15, 0x1c), Panel = RGB(0x1b, 0x20, 0x2b), Field = RGB(0x24, 0x2b, 0x39), TextColor = RGB(0xe8, 0xea, 0xf0),
                   Muted = RGB(0x8d, 0x96, 0xa8), Accent = RGB(0xe0, 0xb6, 0x4a), Good = RGB(0x6f, 0xcf, 0x7f), Bad = RGB(0xef, 0x6b, 0x6b);
constexpr int NeededGb = 16; // (MSYS2 with the build tools, the build's objects, the game's caches)

// The installation's parts: the key install.ps1 / launcher.py / build_game.py name in "@step", the title, the share of the whole.
struct Step { const wchar_t* key; const wchar_t* title; int weight; int state = 0; DWORD started = 0, took = 0; }; // state: 0 waiting, 1 running, 2 done, 3 failed
Step steps[] = {
    {L"msys", L"Set up the build environment (MSYS2)", 6},
    {L"packages", L"Download the build tools", 10},
    {L"source", L"Get TOoT:R's source", 2},
    {L"tools", L"Prepare the translator", 3},
    {L"check", L"Read and check your game", 3},
    {L"translate", L"Translate the game's code", 6},
    {L"compile", L"Compile the game", 55},
    {L"runtime", L"Build TOoT:R", 13},
    {L"shortcuts", L"Create the shortcuts", 2},
};

enum Page { Setup, Working, Done, Failed } page = Setup;
HWND window, nup_edit, dir_edit, log_edit, install_button, close_button, play_button, details_button, back_button, folder_button, nup_browse, dir_browse;
HFONT font, bold_font, title_font, heading_font, mono_font;
HBRUSH bg_brush, field_brush, panel_brush;
HICON icon;
HANDLE child = nullptr;
std::wstring install_dir, doing = L"Starting...", failure, pending_line, nup_note, space_note;
bool nup_ok = false, space_ok = false, details = false;
int current = -1;
double fraction = 0;
DWORD started = 0;
int dpi = 96;

int S(int v) { return MulDiv(v, dpi, 96); }

std::wstring DefaultDir() {
    wchar_t buffer[MAX_PATH];
    const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buffer, MAX_PATH);
    return (n && n < MAX_PATH ? std::wstring(buffer) : std::wstring(L"C:")) + L"\\tootr";
}

std::wstring Text(HWND h) {
    std::wstring s(GetWindowTextLengthW(h), L'\0');
    if (!s.empty()) GetWindowTextW(h, s.data(), (int)s.size() + 1);
    while (!s.empty() && (s.back() == L' ' || s.back() == L'"')) s.pop_back();
    while (!s.empty() && (s.front() == L' ' || s.front() == L'"')) s.erase(s.begin());
    return s;
}

std::wstring Duration(DWORD ms) {
    const DWORD s = ms / 1000;
    return s >= 90 ? std::to_wstring((s + 30) / 60) + L" min" : std::to_wstring(s) + L" s";
}

std::wstring Size(ULONGLONG bytes) {
    wchar_t t[32];
    swprintf(t, 32, bytes >= (1ull << 30) ? L"%.1f GB" : L"%.0f MB", bytes >= (1ull << 30) ? bytes / 1073741824.0 : bytes / 1048576.0);
    return t;
}

// The newest .nup in the folders a dump usually ends up in.
std::wstring FindGame() {
    std::wstring best; FILETIME newest{};
    wchar_t profile[MAX_PATH];
    if (!GetEnvironmentVariableW(L"USERPROFILE", profile, MAX_PATH)) return best;
    for (const wchar_t* folder : {L"\\Downloads", L"\\Desktop", L"\\Documents", L""}) {
        const std::wstring base = std::wstring(profile) + folder + L"\\";
        WIN32_FIND_DATAW found;
        HANDLE h = FindFirstFileW((base + L"*.nup").c_str(), &found);
        if (h == INVALID_HANDLE_VALUE) continue;
        do {
            if (CompareFileTime(&found.ftLastWriteTime, &newest) > 0) { newest = found.ftLastWriteTime; best = base + found.cFileName; }
        } while (FindNextFileW(h, &found));
        FindClose(h);
    }
    return best;
}

// What the Setup page says under its two fields, and whether Install may be pressed.
void Check() {
    const std::wstring nup = Text(nup_edit);
    WIN32_FILE_ATTRIBUTE_DATA info;
    nup_ok = false;
    if (nup.empty()) nup_note = L"Choose your .nup file.";
    else if (!GetFileAttributesExW(nup.c_str(), GetFileExInfoStandard, &info) || (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) nup_note = L"There is no such file.";
    else {
        const ULONGLONG size = (ULONGLONG)info.nFileSizeHigh << 32 | info.nFileSizeLow;
        const std::wstring name = nup.substr(nup.find_last_of(L"\\/") + 1);
        nup_ok = size >= (1ull << 30);
        nup_note = nup_ok ? name + L"  (" + Size(size) + L"). It is checked exactly when the installation starts."
                          : name + L" is too small to be the game (" + Size(size) + L").";
    }
    std::wstring folder = Text(dir_edit);
    if (folder.empty()) folder = DefaultDir();
    std::wstring probe = folder;
    ULARGE_INTEGER free{};
    while (!probe.empty() && !GetDiskFreeSpaceExW(probe.c_str(), &free, nullptr, nullptr)) {
        const size_t cut = probe.find_last_of(L"\\/");
        probe = cut == std::wstring::npos ? L"" : probe.substr(0, cut);
    }
    space_ok = probe.empty() || free.QuadPart >= (ULONGLONG)NeededGb << 30 || GetFileAttributesW((folder + L"\\build").c_str()) != INVALID_FILE_ATTRIBUTES;
    space_note = probe.empty() ? L"" : space_ok ? Size(free.QuadPart) + L" free (about " + std::to_wstring(NeededGb) + L" GB are used while building)."
                                                 : L"Only " + Size(free.QuadPart) + L" free: about " + std::to_wstring(NeededGb) + L" GB are needed while building.";
    EnableWindow(install_button, nup_ok && space_ok);
    InvalidateRect(window, nullptr, TRUE);
}

void BrowseNup() {
    wchar_t path[MAX_PATH] = L"";
    OPENFILENAMEW of{sizeof of};
    of.hwndOwner = window;
    of.lpstrFilter = L"Switch 2 package (*.nup)\0*.nup\0All files\0*.*\0";
    of.lpstrFile = path; of.nMaxFile = MAX_PATH;
    of.lpstrTitle = L"Your dump of the game";
    of.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_EXPLORER;
    if (GetOpenFileNameW(&of)) SetWindowTextW(nup_edit, path);
}

void BrowseDir() {
    BROWSEINFOW bi{};
    bi.hwndOwner = window;
    bi.lpszTitle = L"Install into (a TOoT-R folder is made there)";
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    if (PIDLIST_ABSOLUTE id = SHBrowseForFolderW(&bi)) {
        wchar_t path[MAX_PATH];
        if (SHGetPathFromIDListW(id, path)) {
            std::wstring p = path;
            if (!p.empty() && p.back() != L'\\') p += L'\\';
            SetWindowTextW(dir_edit, (p + L"TOoT-R").c_str());
        }
        CoTaskMemFree(id);
    }
}

void AppendLog(const std::wstring& line) {
    const int length = GetWindowTextLengthW(log_edit);
    if (length > 400000) { SendMessageW(log_edit, EM_SETSEL, 0, length / 2); SendMessageW(log_edit, EM_REPLACESEL, FALSE, (LPARAM)L""); }
    const int end = GetWindowTextLengthW(log_edit);
    SendMessageW(log_edit, EM_SETSEL, end, end);
    SendMessageW(log_edit, EM_REPLACESEL, FALSE, (LPARAM)(line + L"\r\n").c_str());
}

// A part begins: the ones before it are done (also those that had nothing to do).
void Begin(const std::wstring& key) {
    int index = -1;
    const int count = (int)(sizeof steps / sizeof *steps);
    for (int i = 0; i < count; ++i) if (key == steps[i].key) index = i;
    if (index < 0 && key != L"done") return;
    const DWORD now = GetTickCount();
    for (int i = 0; i < (index < 0 ? count : index); ++i) {
        if (steps[i].state == 1) steps[i].took = now - steps[i].started;
        steps[i].state = 2;
    }
    current = index; fraction = 0;
    if (index >= 0) { steps[index].state = 1; steps[index].started = now; }
}

double Overall() {
    double total = 0, done = 0;
    for (const Step& s : steps) { total += s.weight; if (s.state == 2) done += s.weight; else if (s.state == 1) done += s.weight * fraction; }
    return total > 0 ? done / total : 0;
}

// One line of the installation's output.
void ReadLine(std::wstring line) {
    while (!line.empty() && (line.back() == L'\r' || line.back() == L' ')) line.pop_back();
    if (line.empty()) return;
    if (line.rfind(L"@step ", 0) == 0) { Begin(line.substr(6)); return; }
    if (line.rfind(L"@progress ", 0) == 0) {
        const size_t space = line.find(L' ', 10);
        fraction = _wtoi(line.substr(10).c_str()) / 100.0;
        if (space != std::wstring::npos && space + 1 < line.size()) doing = line.substr(space + 1);
        return;
    }
    int n = 0, of = 0, used = 0;
    if (swscanf(line.c_str(), L"[%d/%d] %n", &n, &of, &used) == 2 && of > 0) {
        fraction = (double)n / of;
        std::wstring what = line.substr(used);
        const size_t object = what.find(L".cpp.o");
        if (object != std::wstring::npos) {
            const size_t slash = what.find_last_of(L"/\\", object);
            const wchar_t* where = current >= 0 && !wcscmp(steps[current].key, L"compile") ? L"the game's code" : current >= 0 && !wcscmp(steps[current].key, L"runtime") ? L"TOoT:R" : L"the translator";
            doing = std::wstring(L"Compiling ") + where + L": " + what.substr(slash == std::wstring::npos ? 0 : slash + 1, object - (slash == std::wstring::npos ? 0 : slash + 1)) +
                    L"  (file " + std::to_wstring(n) + L" of " + std::to_wstring(of) + L")";
        } else doing = what.substr(0, 90) + L"  (" + std::to_wstring(n) + L" of " + std::to_wstring(of) + L")";
    } else if (line.rfind(L"[TOoT:R] ", 0) == 0 || line.rfind(L"[build] ", 0) == 0) {
        const std::wstring said = line.substr(line.find(L"] ") + 2);
        if (said.find(L" ...") == std::wstring::npos && said.find(L":\\") != 1 && said.rfind(L"cmake ", 0) != 0) doing = said.substr(0, 120);
        if (said.rfind(L"error: ", 0) == 0) failure = said.substr(7);
    }
    AppendLog(line);
}

// Reads the script's output and hands it to the window.
DWORD WINAPI Reader(LPVOID pipe) {
    char buffer[4096];
    DWORD got;
    while (ReadFile(pipe, buffer, sizeof buffer, &got, nullptr) && got) {
        const int n = MultiByteToWideChar(CP_OEMCP, 0, buffer, (int)got, nullptr, 0);
        auto* text = new std::wstring(n, L'\0');
        MultiByteToWideChar(CP_OEMCP, 0, buffer, (int)got, text->data(), n);
        PostMessageW(window, WmOutput, 0, (LPARAM)text);
    }
    CloseHandle(pipe);
    DWORD code = 1;
    WaitForSingleObject(child, INFINITE);
    GetExitCodeProcess(child, &code);
    PostMessageW(window, WmDone, code, 0);
    return 0;
}

void Show(Page p);

void Install() {
    const std::wstring nup = Text(nup_edit);
    install_dir = Text(dir_edit);
    if (install_dir.empty()) install_dir = DefaultDir();
    // The built-in script, into a folder of its own under %TEMP%.
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const std::wstring folder = std::wstring(temp) + L"tootr-setup";
    CreateDirectoryW(folder.c_str(), nullptr);
    const std::wstring script = folder + L"\\install.ps1";
    HANDLE f = CreateFileW(script.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    DWORD written = 0;
    if (f == INVALID_HANDLE_VALUE || !WriteFile(f, kInstallScript, (DWORD)kInstallScriptSize, &written, nullptr)) {
        if (f != INVALID_HANDLE_VALUE) CloseHandle(f);
        MessageBoxW(window, L"Cannot write the setup script to the temporary folder.", L"TOoT:R", MB_ICONERROR);
        return;
    }
    CloseHandle(f);

    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE read = nullptr, write = nullptr;
    CreatePipe(&read, &write, &sa, 0);
    SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{sizeof si};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = si.hStdError = write;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    std::wstring command = L"powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"" + script +
                           L"\" -Nup \"" + nup + L"\" -Dir \"" + install_dir + L"\"";
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, folder.c_str(), &si, &pi)) {
        CloseHandle(read); CloseHandle(write);
        MessageBoxW(window, L"Cannot start PowerShell.", L"TOoT:R", MB_ICONERROR);
        return;
    }
    CloseHandle(write);
    CloseHandle(pi.hThread);
    child = pi.hProcess;
    for (Step& s : steps) { s.state = 0; s.took = 0; }
    current = -1; fraction = 0; failure.clear(); doing = L"Starting..."; pending_line.clear();
    SetWindowTextW(log_edit, L"");
    started = GetTickCount();
    SetTimer(window, 1, 500, nullptr);
    Show(Working);
    CloseHandle(CreateThread(nullptr, 0, Reader, read, 0, nullptr));
}

void Play() {
    const std::wstring runner = install_dir + L"\\tootr.cmd";
    ShellExecuteW(window, L"open", runner.c_str(), nullptr, install_dir.c_str(), SW_SHOWMINNOACTIVE);
    DestroyWindow(window);
}

HWND Make(const wchar_t* cls, const wchar_t* text, DWORD style, int id, DWORD ex = 0) {
    HWND c = CreateWindowExW(ex, cls, text, WS_CHILD | style, 0, 0, 10, 10, window, (HMENU)(INT_PTR)id, GetModuleHandleW(nullptr), nullptr);
    SendMessageW(c, WM_SETFONT, (WPARAM)font, TRUE);
    return c;
}

int StepsBottom() { return 150 + 26 * (int)(sizeof steps / sizeof *steps); }

void Layout() {
    RECT r; GetClientRect(window, &r);
    const int w = MulDiv(r.right, 96, dpi), h = MulDiv(r.bottom, 96, dpi);
    const auto place = [&](HWND c, int x, int y, int cw, int ch, bool shown) { MoveWindow(c, S(x), S(y), S(cw), S(ch), TRUE); ShowWindow(c, shown ? SW_SHOW : SW_HIDE); };
    const bool setup = page == Setup, working = page == Working || page == Failed;
    place(nup_edit, 26, 172, w - 26 - 150, 30, setup);
    place(nup_browse, w - 116, 172, 90, 30, setup);
    place(dir_edit, 26, 296, w - 26 - 150, 30, setup);
    place(dir_browse, w - 116, 296, 90, 30, setup);
    place(install_button, w - 176, h - 62, 150, 42, setup);
    place(play_button, w - 176, h - 62, 150, 42, page == Done);
    place(close_button, w - (setup || page == Done ? 286 : 126), h - 58, 100, 34, true);
    place(back_button, w - 236, h - 58, 100, 34, page == Failed);
    place(folder_button, 26, h - 58, 150, 34, page == Done);
    place(details_button, 26, h - 58, 130, 34, working);
    const int top = StepsBottom() + 74;
    place(log_edit, 26, top, w - 52, h - top - 74, working && details && h - top - 74 > 30);
    SetWindowTextW(close_button, page == Working ? L"Cancel" : L"Close");
    SetWindowTextW(details_button, details ? L"Hide details" : L"Show details");
}

void Show(Page p) {
    page = p;
    if (p == Failed) details = true;
    Layout();
    InvalidateRect(window, nullptr, TRUE);
}

void Draw(HDC dc, const std::wstring& text, int x, int y, int w, COLORREF color, HFONT f, UINT format = DT_LEFT | DT_WORDBREAK | DT_NOPREFIX) {
    RECT r{S(x), S(y), S(x + w), S(y + 400)};
    SelectObject(dc, f); SetTextColor(dc, color); SetBkMode(dc, TRANSPARENT);
    DrawTextW(dc, text.c_str(), (int)text.size(), &r, format | DT_NOCLIP);
}

void Fill(HDC dc, int x, int y, int w, int h, COLORREF color) {
    RECT r{S(x), S(y), S(x + w), S(y + h)};
    HBRUSH b = CreateSolidBrush(color); FillRect(dc, &r, b); DeleteObject(b);
}

// A step's mark, drawn (fonts differ in which symbols they have): 0 a ring, 1 a filled dot, 2 a tick, 3 a cross.
void Mark(HDC dc, int x, int y, int state) {
    const COLORREF color = state == 2 ? Good : state == 1 ? Accent : state == 3 ? Bad : Muted;
    HPEN pen = CreatePen(PS_SOLID, S(2), color); HBRUSH fill = CreateSolidBrush(state == 1 ? color : Bg);
    HGDIOBJ old_pen = SelectObject(dc, pen), old_brush = SelectObject(dc, fill);
    if (state == 2) { POINT p[] = {{S(x + 1), S(y + 9)}, {S(x + 5), S(y + 13)}, {S(x + 13), S(y + 4)}}; Polyline(dc, p, 3); }
    else if (state == 3) { MoveToEx(dc, S(x + 2), S(y + 3), nullptr); LineTo(dc, S(x + 12), S(y + 13)); MoveToEx(dc, S(x + 12), S(y + 3), nullptr); LineTo(dc, S(x + 2), S(y + 13)); }
    else Ellipse(dc, S(x + 2), S(y + 3), S(x + 13), S(y + 14));
    SelectObject(dc, old_pen); SelectObject(dc, old_brush); DeleteObject(pen); DeleteObject(fill);
}

void Paint(HDC dc) {
    RECT client; GetClientRect(window, &client);
    const int w = MulDiv(client.right, 96, dpi), h = MulDiv(client.bottom, 96, dpi);
    FillRect(dc, &client, bg_brush);
    Fill(dc, 0, 0, w, 96, Panel);
    if (icon) DrawIconEx(dc, S(26), S(16), icon, S(64), S(64), 0, nullptr, DI_NORMAL);
    Draw(dc, L"TOoT:R", 106, 14, 400, Accent, title_font);
    Draw(dc, L"A native PC version of the game, built on this computer from your own copy.", 106, 58, w - 130, Muted, font);
    if (page == Setup) {
        Draw(dc, L"1   Your copy of the game", 26, 112, 500, Accent, heading_font);
        Draw(dc, L"The Legend of Zelda: Ocarina of Time for Nintendo Switch 2, update v131072, dumped as one .nup file. It is only read on this computer: nothing from it is uploaded or downloaded.",
             26, 134, w - 52, Muted, font);
        Draw(dc, nup_note, 26, 206, w - 52, nup_ok ? Good : Text(nup_edit).empty() ? Muted : Bad, font);
        Draw(dc, L"2   Where to install", 26, 246, 500, Accent, heading_font);
        Draw(dc, L"The game, its saves and its settings all live in this folder. Leave it empty for the default.", 26, 268, w - 52, Muted, font);
        Draw(dc, space_note, 26, 330, w - 52, space_ok ? Good : Bad, font);
        Draw(dc, L"3   Build tools", 26, 370, 500, Accent, heading_font);
        Draw(dc, L"MSYS2 (compilers and libraries, about 3 GB) is downloaded and set up in C:\\msys64 if it is not there yet. No administrator rights are needed.", 26, 392, w - 52, Muted, font);
        Draw(dc, L"The first installation compiles the whole game:\n15 to 60 minutes, depending on the processor.", 26, h - 62, 420, Muted, font);
        return;
    }
    if (page == Done) {
        Mark(dc, 28, 126, 2);
        Draw(dc, L"TOoT:R is installed", 52, 116, w - 78, Good, title_font);
        Draw(dc, L"It took " + Duration(GetTickCount() - started) + L". Updates are checked for, and built, each time the game starts.", 26, 160, w - 52, Muted, font);
        Fill(dc, 26, 196, w - 52, 150, Panel);
        const std::pair<const wchar_t*, std::wstring> tips[] = {
            {L"Start it", L"from the Start menu or the desktop shortcut (TOoT-R)."},
            {L"Settings", L"F1 or Insert in the game opens them: graphics, controller, keyboard and mouse. On a controller, press Select and Start together."},
            {L"Installed in", install_dir},
        };
        int y = 208;
        for (const auto& [title, text] : tips) {
            Draw(dc, title, 42, y, 110, Accent, bold_font);
            Draw(dc, text, 160, y, w - 52 - 150, TextColor, font);
            y += text.size() > 80 ? 52 : 34;
        }
        return;
    }
    Draw(dc, page == Failed ? L"The installation stopped" : L"Installing", 26, 108, 500, page == Failed ? Bad : TextColor, heading_font);
    Draw(dc, L"Into " + install_dir, 26, 130, w - 52, Muted, font, DT_LEFT | DT_SINGLELINE | DT_PATH_ELLIPSIS | DT_NOPREFIX);
    int y = 156;
    for (const Step& s : steps) {
        Mark(dc, 28, y, s.state);
        Draw(dc, s.title, 52, y, w - 200, s.state ? TextColor : Muted, s.state == 1 ? bold_font : font);
        if (s.state == 2 && s.took > 1500) Draw(dc, Duration(s.took), w - 146, y, 120, Muted, font, DT_RIGHT | DT_SINGLELINE);
        y += 26;
    }
    const double overall = Overall();
    Fill(dc, 26, y + 8, w - 52, 16, Field);
    Fill(dc, 26, y + 8, (int)((w - 52) * overall), 16, page == Failed ? Bad : Accent);
    Draw(dc, std::to_wstring((int)(overall * 100)) + L"%", 26, y + 30, 100, TextColor, bold_font);
    const DWORD elapsed = GetTickCount() - started;
    if (page == Working)
        Draw(dc, (overall > 0.08 && elapsed > 45000 ? L"about " + Duration((DWORD)(elapsed * (1 - overall) / overall)) + L" left, " : L"") + Duration(elapsed) + L" so far",
             w - 426, y + 30, 400, Muted, font, DT_RIGHT | DT_SINGLELINE);
    Draw(dc, page == Failed ? (failure.empty() ? L"See the details below for what went wrong." : failure) : doing, 26, y + 52, w - 52, page == Failed ? Bad : Muted, font,
         DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX);
}

// A flat button: the accent colour for the page's main one.
void DrawButton(const DRAWITEMSTRUCT* d) {
    const bool main = d->hwndItem == install_button || d->hwndItem == play_button;
    const bool disabled = d->itemState & ODS_DISABLED, pressed = d->itemState & ODS_SELECTED;
    const COLORREF back = main ? (disabled ? RGB(0x5a, 0x4d, 0x2a) : pressed ? RGB(0xc8, 0xa0, 0x3c) : Accent) : (pressed ? RGB(0x32, 0x3b, 0x4d) : Field);
    HBRUSH b = CreateSolidBrush(back); FillRect(d->hDC, &d->rcItem, b); DeleteObject(b);
    wchar_t text[64]; GetWindowTextW(d->hwndItem, text, 64);
    SelectObject(d->hDC, main ? bold_font : font);
    SetBkMode(d->hDC, TRANSPARENT);
    SetTextColor(d->hDC, main ? RGB(0x1a, 0x14, 0x07) : disabled ? Muted : TextColor);
    RECT r = d->rcItem;
    DrawTextW(d->hDC, text, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    if (d->itemState & ODS_FOCUS) { InflateRect(&r, -S(3), -S(3)); DrawFocusRect(d->hDC, &r); }
}

LRESULT CALLBACK Proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        window = h;
        nup_edit = Make(L"EDIT", L"", WS_TABSTOP | ES_AUTOHSCROLL, IdNup);
        nup_browse = Make(L"BUTTON", L"Browse...", WS_TABSTOP | BS_OWNERDRAW, IdNupBrowse);
        dir_edit = Make(L"EDIT", L"", WS_TABSTOP | ES_AUTOHSCROLL, IdDir);
        SendMessageW(dir_edit, EM_SETCUEBANNER, TRUE, (LPARAM)(L"Default: " + DefaultDir()).c_str());
        dir_browse = Make(L"BUTTON", L"Change...", WS_TABSTOP | BS_OWNERDRAW, IdDirBrowse);
        log_edit = Make(L"EDIT", L"", WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL, IdLog);
        SendMessageW(log_edit, WM_SETFONT, (WPARAM)mono_font, TRUE);
        SendMessageW(log_edit, EM_SETLIMITTEXT, 0, 0);
        details_button = Make(L"BUTTON", L"Show details", WS_TABSTOP | BS_OWNERDRAW, IdDetails);
        folder_button = Make(L"BUTTON", L"Open the folder", WS_TABSTOP | BS_OWNERDRAW, IdFolder);
        back_button = Make(L"BUTTON", L"Back", WS_TABSTOP | BS_OWNERDRAW, IdBack);
        close_button = Make(L"BUTTON", L"Close", WS_TABSTOP | BS_OWNERDRAW, IdClose);
        install_button = Make(L"BUTTON", L"Install", WS_TABSTOP | BS_OWNERDRAW, IdInstall);
        play_button = Make(L"BUTTON", L"Play", WS_TABSTOP | BS_OWNERDRAW, IdPlay);
        const std::wstring game = FindGame();
        if (!game.empty()) SetWindowTextW(nup_edit, game.c_str());
        Layout();
        Check();
        return 0;
    }
    case WM_SIZE: if (nup_edit) { Layout(); InvalidateRect(h, nullptr, TRUE); } return 0;
    case WM_GETMINMAXINFO: { auto* m = (MINMAXINFO*)lp; m->ptMinTrackSize = {S(760), S(600)}; return 0; }
    case WM_ERASEBKGND: return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
        RECT r; GetClientRect(h, &r);
        HDC memory = CreateCompatibleDC(dc); HBITMAP bitmap = CreateCompatibleBitmap(dc, r.right, r.bottom); // (no flicker while the progress moves)
        HGDIOBJ old = SelectObject(memory, bitmap);
        Paint(memory);
        BitBlt(dc, 0, 0, r.right, r.bottom, memory, 0, 0, SRCCOPY);
        SelectObject(memory, old); DeleteObject(bitmap); DeleteDC(memory);
        EndPaint(h, &ps);
        return 0;
    }
    case WM_CTLCOLOREDIT: case WM_CTLCOLORSTATIC: {
        const bool log = (HWND)lp == log_edit;
        SetTextColor((HDC)wp, log ? Muted : TextColor); SetBkColor((HDC)wp, log ? Panel : Field);
        return (LRESULT)(log ? panel_brush : field_brush);
    }
    case WM_DRAWITEM: DrawButton((const DRAWITEMSTRUCT*)lp); return TRUE;
    case WM_TIMER: if (page == Working) { RECT r; GetClientRect(h, &r); r.top = S(100); r.bottom = S(StepsBottom() + 74); InvalidateRect(h, &r, FALSE); } return 0;
    case WM_COMMAND:
        if (HIWORD(wp) == EN_CHANGE && (LOWORD(wp) == IdNup || LOWORD(wp) == IdDir)) { Check(); return 0; }
        switch (LOWORD(wp)) {
        case IdNupBrowse: BrowseNup(); return 0;
        case IdDirBrowse: BrowseDir(); return 0;
        case IdInstall: Install(); return 0;
        case IdPlay: Play(); return 0;
        case IdDetails: details = !details; Layout(); InvalidateRect(h, nullptr, TRUE); return 0;
        case IdBack: Show(Setup); Check(); return 0;
        case IdFolder: ShellExecuteW(h, L"open", install_dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL); return 0;
        case IdClose: SendMessageW(h, WM_CLOSE, 0, 0); return 0;
        }
        break;
    case WmOutput: {
        auto* text = (std::wstring*)lp;
        pending_line += *text; delete text;
        size_t at;
        while ((at = pending_line.find(L'\n')) != std::wstring::npos) { ReadLine(pending_line.substr(0, at)); pending_line.erase(0, at + 1); }
        return 0;
    }
    case WmDone:
        if (!pending_line.empty()) { ReadLine(pending_line); pending_line.clear(); }
        KillTimer(h, 1);
        CloseHandle(child); child = nullptr;
        if (wp == 0) { Begin(L"done"); Show(Done); }
        else { if (current >= 0) steps[current].state = 3; Show(Failed); }
        FlashWindow(h, TRUE);
        return 0;
    case WM_CLOSE:
        if (child) {
            if (MessageBoxW(h, L"Stop the installation?\n\nWhat is built so far is kept: installing again carries on from there.", L"TOoT:R", MB_YESNO | MB_ICONQUESTION) != IDYES) return 0;
            TerminateProcess(child, 1);
        }
        DestroyWindow(h);
        return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    using SetContext = BOOL(WINAPI*)(HANDLE);
    if (auto set = (SetContext)GetProcAddress(GetModuleHandleW(L"user32.dll"), "SetProcessDpiAwarenessContext"))
        set((HANDLE)-4); // (DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    else SetProcessDPIAware();
    INITCOMMONCONTROLSEX icc{sizeof icc, ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&icc);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    HDC screen = GetDC(nullptr); dpi = GetDeviceCaps(screen, LOGPIXELSY); ReleaseDC(nullptr, screen);
    NONCLIENTMETRICSW ncm{sizeof ncm};
    SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof ncm, &ncm, 0);
    LOGFONTW base = ncm.lfMessageFont;
    base.lfHeight = -S(14); font = CreateFontIndirectW(&base);
    LOGFONTW f = base; f.lfWeight = FW_BOLD; bold_font = CreateFontIndirectW(&f);
    f = base; f.lfHeight = -S(17); f.lfWeight = FW_BOLD; heading_font = CreateFontIndirectW(&f);
    f = base; f.lfHeight = -S(28); f.lfWeight = FW_BOLD; title_font = CreateFontIndirectW(&f);
    f = base; f.lfHeight = -S(12); wcscpy(f.lfFaceName, L"Consolas"); mono_font = CreateFontIndirectW(&f);
    bg_brush = CreateSolidBrush(Bg); field_brush = CreateSolidBrush(Field); panel_brush = CreateSolidBrush(Panel);
    icon = (HICON)LoadImageW(instance, MAKEINTRESOURCEW(1), IMAGE_ICON, S(64), S(64), 0);

    WNDCLASSEXW wc{sizeof wc};
    wc.lpfnWndProc = Proc; wc.hInstance = instance; wc.lpszClassName = L"TootrSetup";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = bg_brush;
    wc.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(1));
    wc.hIconSm = wc.hIcon;
    RegisterClassExW(&wc);
    RECT r{0, 0, S(880), S(680)};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    CreateWindowExW(0, L"TootrSetup", L"TOoT:R Installer", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT,
                    r.right - r.left, r.bottom - r.top, nullptr, nullptr, instance, nullptr);
    if (BOOL dark = TRUE; true) { // (a dark title bar where Windows has one: DWMWA_USE_IMMERSIVE_DARK_MODE)
        if (HMODULE dwm = LoadLibraryW(L"dwmapi.dll")) {
            using SetAttribute = HRESULT(WINAPI*)(HWND, DWORD, LPCVOID, DWORD);
            if (auto set = (SetAttribute)GetProcAddress(dwm, "DwmSetWindowAttribute")) set(window, 20, &dark, sizeof dark);
        }
    }
    // (development: TOOTR_SETUP_REPLAY=<a saved installation log> shows the installation page as that log leaves it,
    // or the last page when the log ends with "@step done")
    wchar_t replay[MAX_PATH];
    if (const DWORD n = GetEnvironmentVariableW(L"TOOTR_SETUP_REPLAY", replay, MAX_PATH); n && n < MAX_PATH) {
        HANDLE f = CreateFileW(replay, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (f != INVALID_HANDLE_VALUE) {
            install_dir = DefaultDir(); started = GetTickCount() - 7 * 60 * 1000;
            std::string bytes(GetFileSize(f, nullptr), '\0'); DWORD got = 0;
            ReadFile(f, bytes.data(), (DWORD)bytes.size(), &got, nullptr); CloseHandle(f);
            std::wstring text(bytes.begin(), bytes.begin() + got), line;
            bool done = false;
            for (wchar_t ch : text) { if (ch != L'\n') { line += ch; continue; } done = line.rfind(L"@step done", 0) == 0; ReadLine(line); line.clear(); }
            if (done) Begin(L"done");
            Show(done ? Done : Working);
        }
    }
    ShowWindow(window, show);
    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0)) {
        if (IsDialogMessageW(window, &m)) continue;
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    return 0;
}
