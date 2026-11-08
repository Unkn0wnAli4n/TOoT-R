# TOoT:R (The Ocarina of Time: Recomp)

A native PC port of **The Legend of Zelda: Ocarina of Time** (Nintendo Switch 2 release), made by
static recompilation: the game's AArch64 code is translated to C++ and compiled for x86-64, and the
console's SDK and NVN graphics API are reimplemented on SDL3, Vulkan and FFmpeg. No emulator runs.

**This repository contains no game code or assets.** The installer builds the port from your own
legally obtained dump of the game, on your machine.

## Requirements

- An x86-64 PC with a Vulkan 1.3 GPU and driver supporting `VK_EXT_external_memory_host`
  (current NVIDIA and AMD drivers do). 16 GB of RAM recommended for the build.
- Linux, or Windows 10 1809+ / Windows 11.
- Your dump of the game as a single **`.nup`** file containing update **v131072**
  (title ID `0400486024DEC000`). The installer checks the extracted code against known hashes and
  stops if the version does not match.
- About 3 GB of free space while building, ~1 GB afterwards (plus the game file, which stays where
  it is).

The first build compiles the whole game and takes roughly 15–60 minutes depending on the CPU.

## Install

Download the installer for your system from the
[latest release](https://github.com/Unkn0wnAli4n/TOoT-R/releases/latest):

- **Linux:** `TOoT-R-Installer-x86_64.AppImage`. Make it executable (`chmod +x`, or Properties →
  Permissions) and open it.
- **Windows:** `TOoT-R-Installer.exe`.

The installer asks for your game file and the folder to install into (leave it empty for the default:
`~/.local/share/tootr` on Linux, `%LOCALAPPDATA%\tootr` on Windows), installs the build tools
(Linux: your package manager, pacman, apt, dnf or zypper, after asking for your password; Windows:
MSYS2), builds the game there, and adds a `tootr` command and an application menu entry (Linux) or
Start menu and desktop shortcuts (Windows). The game, its saves and settings all live in the chosen
folder. Distributions without SDL3 packages need SDL3 installed first.

From a clone of this repository the same installation runs in a terminal: `./install.sh` (Linux,
`--nup FILE`, `--dir FOLDER`, `--no-deps`) or `install.cmd` (Windows).

## Playing

Start **TOoT:R** from the menu, or run `tootr`. Controls: any SDL
gamepad, or the keyboard (WASD move, arrows D-pad, J/K/U/I = A/B/X/Y, Q/E = L/R, Shift/Space =
ZL/ZR, Enter = +, Backspace = −). **Insert** shows or hides the graphics settings window (mouse,
keyboard or gamepad).

Data (saves, settings, shader caches) lives in the install folder.

## Updates

Each start checks this repository for a newer version. When there is one, the launcher downloads
it and rebuilds before starting. Usually only the runtime is rebuilt (a minute or two). The full
game code is recompiled only when the recompiler changes. If an update fails to build, the
installed version keeps running. Set `OOT_RECOMP_NO_UPDATE=1` to skip the check. To update by
hand, run `python3 ~/.local/share/tootr/source/installer/launcher.py update`.

To point the install at a different copy of the game, run `launcher.py set-game`.

## Development

```sh
tools/dev.sh            # run the last build (builds first if there is none)
tools/dev.sh build      # (re)build from this working tree into .dev/
tools/dev.sh build-run  # (re)build, then run
```

`tools/build_game.py --help` describes the pipeline. The installers are built with
`tools/package/appimage.sh` (Linux) and `tools/package/windows.sh` (Windows, cross-compiled with clang;
`MINGW_SYSROOT` names a MinGW-w64 sysroot such as MSYS2's `clang64`).

## Status

The game boots, renders, and is playable with a controller. Performance and some rendering details
are still in progress. The Windows port is new; please report issues.

## Credits

Built on [dynarmic](https://github.com/merryhime/dynarmic) (A64 decoder), the
[Eden](https://git.eden-emu.dev/eden-emu/eden)/yuzu shader recompiler, [sirit](https://github.com/ReinUsesLisp/sirit),
[Hookshot](https://github.com/NotRexed/Hookshot) (audio),
[zbic](https://github.com/kinnay/zbic)/[Atmosphère](https://github.com/Atmosphere-NX/Atmosphere) (ZBIC
decompression), AMD [FidelityFX FSR 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR) and
[Dear ImGui](https://github.com/ocornut/imgui).

## License

GPL-3.0-or-later (see `LICENSE`). Third-party code under `third_party/` keeps its own license; see
the PROVENANCE/LICENSE files there.
