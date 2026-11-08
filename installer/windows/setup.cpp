// TOoT:R installer for Windows (TOoT-R-Installer.exe; built by tools/package/windows.sh).
//
// A window that asks for your own dump of the game (.nup) and the folder to install into (blank: the
// default, %LOCALAPPDATA%\tootr), then runs the built-in install.ps1 (MSYS2 build tools, the build,
// shortcuts) with its output shown in the window.
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

#include "setup_script.h" // kInstallScript, kInstallScriptSize (install.ps1)

namespace {

enum : int { IdNup = 101, IdNupBrowse, IdDir, IdDirBrowse, IdLog, IdInstall, IdClose, IdPlay, IdProgress };
constexpr UINT WmOutput = WM_APP + 1, WmDone = WM_APP + 2;

HWND window, nup_edit, dir_edit, log_edit, install_button, play_button, progress;
HFONT font, title_font;
HANDLE child = nullptr;
std::wstring install_dir;
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

void Append(const std::wstring& text) {
    std::wstring t;
    for (wchar_t c : text) { if (c == L'\n' && (t.empty() || t.back() != L'\r')) t += L'\r'; t += c; }
    const int length = GetWindowTextLengthW(log_edit);
    if (length > 400000) { SendMessageW(log_edit, EM_SETSEL, 0, length / 2); SendMessageW(log_edit, EM_REPLACESEL, FALSE, (LPARAM)L""); }
    const int end = GetWindowTextLengthW(log_edit);
    SendMessageW(log_edit, EM_SETSEL, end, end);
    SendMessageW(log_edit, EM_REPLACESEL, FALSE, (LPARAM)t.c_str());
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

// Reads the script's output and hands it to the window (UTF-8 or the console code page).
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

void Install() {
    const std::wstring nup = Text(nup_edit);
    if (nup.empty() || GetFileAttributesW(nup.c_str()) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(window, L"Choose your .nup file first.", L"TOoT:R", MB_ICONWARNING);
        return;
    }
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
    EnableWindow(install_button, FALSE);
    SendMessageW(progress, PBM_SETMARQUEE, TRUE, 30);
    Append(L"Installing into " + install_dir + L"\n(the first build takes 15-60 minutes)\n\n");
    CloseHandle(CreateThread(nullptr, 0, Reader, read, 0, nullptr));
}

void Play() {
    const std::wstring runner = install_dir + L"\\tootr.cmd";
    ShellExecuteW(window, L"open", runner.c_str(), nullptr, install_dir.c_str(), SW_SHOWMINNOACTIVE);
    DestroyWindow(window);
}

HWND Make(const wchar_t* cls, const wchar_t* text, DWORD style, int x, int y, int w, int h, int id, DWORD ex = 0) {
    HWND c = CreateWindowExW(ex, cls, text, WS_CHILD | WS_VISIBLE | style, S(x), S(y), S(w), S(h), window,
                             (HMENU)(INT_PTR)id, GetModuleHandleW(nullptr), nullptr);
    SendMessageW(c, WM_SETFONT, (WPARAM)font, TRUE);
    return c;
}

void Layout() {
    RECT r; GetClientRect(window, &r);
    const int w = MulDiv(r.right, 96, dpi), h = MulDiv(r.bottom, 96, dpi);
    MoveWindow(nup_edit, S(190), S(78), S(w - 190 - 120), S(24), TRUE);
    MoveWindow(GetDlgItem(window, IdNupBrowse), S(w - 112), S(77), S(96), S(26), TRUE);
    MoveWindow(dir_edit, S(190), S(112), S(w - 190 - 120), S(24), TRUE);
    MoveWindow(GetDlgItem(window, IdDirBrowse), S(w - 112), S(111), S(96), S(26), TRUE);
    MoveWindow(log_edit, S(16), S(170), S(w - 32), S(h - 170 - 84), TRUE);
    MoveWindow(progress, S(16), S(h - 74), S(w - 32), S(14), TRUE);
    MoveWindow(GetDlgItem(window, IdClose), S(w - 336), S(h - 48), S(100), S(30), TRUE);
    MoveWindow(install_button, S(w - 226), S(h - 48), S(100), S(30), TRUE);
    MoveWindow(play_button, S(w - 116), S(h - 48), S(100), S(30), TRUE);
}

LRESULT CALLBACK Proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        window = h;
        HWND title = Make(L"STATIC", L"TOoT:R", 0, 16, 10, 400, 34, 0);
        SendMessageW(title, WM_SETFONT, (WPARAM)title_font, TRUE);
        Make(L"STATIC", L"Builds a native PC version of the game from your own copy. Nothing from the game is downloaded or uploaded.",
             0, 16, 46, 760, 22, 0);
        Make(L"STATIC", L"Game file (.nup, v131072):", 0, 16, 81, 170, 22, 0);
        nup_edit = Make(L"EDIT", L"", WS_TABSTOP | ES_AUTOHSCROLL, 190, 78, 480, 24, IdNup, WS_EX_CLIENTEDGE);
        Make(L"BUTTON", L"Browse...", WS_TABSTOP, 680, 77, 96, 26, IdNupBrowse);
        Make(L"STATIC", L"Install folder:", 0, 16, 115, 170, 22, 0);
        dir_edit = Make(L"EDIT", L"", WS_TABSTOP | ES_AUTOHSCROLL, 190, 112, 480, 24, IdDir, WS_EX_CLIENTEDGE);
        SendMessageW(dir_edit, EM_SETCUEBANNER, TRUE, (LPARAM)(L"Default: " + DefaultDir()).c_str());
        Make(L"BUTTON", L"Browse...", WS_TABSTOP, 680, 111, 96, 26, IdDirBrowse);
        Make(L"STATIC", L"Leave the folder empty for the default. MSYS2 (the build tools) is installed if missing.", 0, 190, 140, 590, 22, 0);
        log_edit = Make(L"EDIT", L"", WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL, 16, 170, 760, 300, IdLog, WS_EX_CLIENTEDGE);
        SendMessageW(log_edit, EM_SETLIMITTEXT, 0, 0);
        progress = Make(PROGRESS_CLASSW, L"", PBS_MARQUEE, 16, 480, 760, 14, IdProgress);
        Make(L"BUTTON", L"Close", WS_TABSTOP, 456, 506, 100, 30, IdClose);
        install_button = Make(L"BUTTON", L"Install", WS_TABSTOP | BS_DEFPUSHBUTTON, 566, 506, 100, 30, IdInstall);
        play_button = Make(L"BUTTON", L"Play", WS_TABSTOP | WS_DISABLED, 676, 506, 100, 30, IdPlay);
        Layout();
        return 0;
    }
    case WM_SIZE: if (nup_edit) Layout(); return 0;
    case WM_GETMINMAXINFO: { auto* m = (MINMAXINFO*)lp; m->ptMinTrackSize = {S(720), S(480)}; return 0; }
    case WM_CTLCOLORSTATIC: if ((HWND)lp == log_edit) { SetBkColor((HDC)wp, GetSysColor(COLOR_WINDOW)); return (LRESULT)GetSysColorBrush(COLOR_WINDOW); } break;
    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case IdNupBrowse: BrowseNup(); return 0;
        case IdDirBrowse: BrowseDir(); return 0;
        case IdInstall: Install(); return 0;
        case IdPlay: Play(); return 0;
        case IdClose: SendMessageW(h, WM_CLOSE, 0, 0); return 0;
        }
        break;
    case WmOutput: { auto* text = (std::wstring*)lp; Append(*text); delete text; return 0; }
    case WmDone:
        SendMessageW(progress, PBM_SETMARQUEE, FALSE, 0);
        CloseHandle(child); child = nullptr;
        EnableWindow(install_button, TRUE);
        if (wp == 0) {
            EnableWindow(play_button, TRUE);
            Append(L"\nInstalled. Start TOoT:R from the Start menu or the desktop shortcut.\n");
            MessageBoxW(h, L"Installed. Start TOoT:R from the Start menu or the desktop shortcut.", L"TOoT:R", MB_ICONINFORMATION);
        } else {
            Append(L"\nThe installation failed: see the messages above.\n");
            MessageBoxW(h, L"The installation failed: see the messages in the window.", L"TOoT:R", MB_ICONERROR);
        }
        return 0;
    case WM_CLOSE:
        if (child) {
            if (MessageBoxW(h, L"Stop the installation?", L"TOoT:R", MB_YESNO | MB_ICONQUESTION) != IDYES) return 0;
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
    INITCOMMONCONTROLSEX icc{sizeof icc, ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&icc);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    HDC screen = GetDC(nullptr); dpi = GetDeviceCaps(screen, LOGPIXELSY); ReleaseDC(nullptr, screen);
    NONCLIENTMETRICSW ncm{sizeof ncm};
    SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof ncm, &ncm, 0);
    ncm.lfMessageFont.lfHeight = -S(14);
    font = CreateFontIndirectW(&ncm.lfMessageFont);
    LOGFONTW big = ncm.lfMessageFont; big.lfHeight = -S(26); big.lfWeight = FW_BOLD;
    title_font = CreateFontIndirectW(&big);

    WNDCLASSEXW wc{sizeof wc};
    wc.lpfnWndProc = Proc; wc.hInstance = instance; wc.lpszClassName = L"TootrSetup";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = GetSysColorBrush(COLOR_BTNFACE);
    wc.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(1));
    wc.hIconSm = wc.hIcon;
    RegisterClassExW(&wc);
    RECT r{0, 0, S(800), S(560)};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    CreateWindowExW(0, L"TootrSetup", L"TOoT:R Installer", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                    r.right - r.left, r.bottom - r.top, nullptr, nullptr, instance, nullptr);
    ShowWindow(window, show);
    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0)) {
        if (IsDialogMessageW(window, &m)) continue;
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    return 0;
}
