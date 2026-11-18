# TOoT:R (The Ocarina of Time: Recomp)

A native PC port of **The Legend of Zelda: Ocarina of Time** (Nintendo Switch 2 release), made by
static recompilation: the game's AArch64 code is translated to C++ and compiled for x86-64, and the
console's system functions (files, input, sound, video, threads) and its graphics API are
reimplemented on SDL3, Vulkan and FFmpeg. No emulator runs.

**This repository contains no game code or assets.** The installer builds the port from your own
legally obtained dump of the game, on your machine.

## What it can do

- **Upscalers.** One *Upscaler* setting chooses what makes the picture: the game's own neural
  network, **NVIDIA DLSS** (DLAA at the Native quality step, with a choice of DLSS model),
  **AMD FSR 3.1** on any GPU, **AMD FSR 4** (experimental; it uses AMD's FSR 4 library for Vulkan
  when your PC has one), or none, with or without FXAA. One Render Resolution setting carries the
  quality steps (100% is DLAA, then Quality, Balanced, Performance, Ultra Performance), and there is
  a sharpness control.
- **Output resolution.** The game's own 1440p picture, or 1080p, 1800p or **4K**.
- **Frame generation.** 2x and more, with NVIDIA DLSS Frame Generation where it runs and AMD
  FSR 3.1's everywhere else. It never makes more frames than your display shows, and the FPS counter
  shows the rate you see.
- **High frame rates.** The game runs above 60 frames per second at the right speed: caps from 15
  to 240, or uncapped. Game logic that counts whole 60 Hz frames (sound, the day's clock, water,
  songs, physics steps) is kept right at every rate. VSync on, off or adaptive (for variable refresh rate displays).
- **Displays.** Windowed or fullscreen; the game's 16:9 picture, or one drawn for 4:3, 16:10, 21:9,
  32:9 or your display's own shape, with the HUD at the screen's edges or kept in the middle. **HDR** output
  (experimental) with peak brightness, paper white and highlight controls.
- **Detail settings.** Presets from Low to Ultra and the original look; shadow quality, foliage
  shadows, texture filtering, texture detail and more, each explained in the settings window.
- **Controls.** Any controller with rumble and **gyro aiming**; keyboard and mouse with a direct
  mouse camera and the **mouse as a pointer in the game's menus**. Every control can be rebound.
- **Game options.** Field of view, camera speed, text speed, skipping the opening.
- **Cheats and hacks.** Infinite health, rupees and magic, damage taken from none to quadruple, autosave
  off; game speed and movement speed. In play, the Cheats tab also has **teleport** to 31 places,
  **set the time of day**, **switch Link's age**, an **inventory editor**, **flight** (no clip), a
  **free camera**, and mark / return to a place.
- **Mods.** RomFS mods in the layout emulators use work as they are: put a mod in
  `<install folder>/mods/<name>/romfs/...` and its files replace or add to the game's. Folders whose
  names start with `-` or `_` are off; where two mods have the same file, the one sorting last wins.
  (Code patches, `exefs`, cannot apply to a recompiled game.)
- **Language and text.** The game's 16 languages, and larger dialogue text (up to 175%).
- **Low latency.** NVIDIA Reflex, or AMD Anti-Lag 2 where the driver has it.
- **FPS counter** with a frame time graph and details (GPU and renderer time, video memory), in any
  corner. A choice of graphics card and sound output on PCs with several.
- **Game files.** Compressed `.nup` packages are read as well as uncompressed ones.

NVIDIA DLSS is not part of this project and is not distributed with it. When you tick the DLSS
option, the installer downloads NVIDIA's DLSS files from NVIDIA's own repository onto your PC, under
NVIDIA's licence, which you accept there. Without it everything else works the same. On Windows
DLSS is new and has not been tried on many PCs yet; tell us how it runs.

## Steam Deck: work in progress

The Steam Deck build starts and plays, but it is **not finished**: the frame rate is not yet stable
in the heavier areas, and long sessions can still stutter or stop. Expect problems, and please
report what you find. PC (Linux and Windows) is the supported way to play for now.

## Requirements

- An x86-64 PC with a Vulkan 1.3 GPU and driver supporting `VK_EXT_external_memory_host`
  (current NVIDIA and AMD drivers do). 16 GB of RAM recommended for the build.
- Linux, or Windows 10 1809+ / Windows 11.
- Your dump of the game as a single **`.nup`** file (title ID `0400486024DEC000`). Update **v131072**
  is the version this port is made and tested with. Other updates are not refused: the build finds
  what it needs in the game's own code by what that code looks like, and leaves out the options whose
  code a version does not have (the build's log names them). An untested update may still not run.
- About 10 GB of free space while building (Windows: 16 GB, with MSYS2), ~3 GB afterwards (plus the
  game file, which stays where it is).

The first build compiles the whole game and takes roughly 15–60 minutes depending on the CPU.

## Install

Download the installer for your system from the
[latest release](https://github.com/Unkn0wnAli4n/TOoT-R/releases/latest):

Join Discord for support:
[discord](https://discord.gg/6j7FJ282np)

- **Linux (and Steam Deck, work in progress):** `TOoT-R-Installer-x86_64.AppImage`. Make it executable (`chmod +x`, or
  Properties → Permissions) and open it.
- **Windows:** `TOoT-R-Installer.exe`.

The installer finds or asks for your game file and the folder to install into (default:
`~/.local/share/tootr` on Linux, `%LOCALAPPDATA%\tootr` on Windows), then shows each step of the
installation with its progress, the file being compiled and the time left. When it is done it adds a
`tootr` command and an application menu entry (Linux) or Start menu and desktop shortcuts (Windows).
The game, its saves and settings all live in the chosen folder.

**Linux needs nothing installed.** The AppImage carries its own build environment (compilers, CMake,
Ninja, Python and the libraries the game uses), unpacks it into the install folder and builds inside
it in a sandbox: no packages and no password, so it also installs on systems whose own files are
read-only (SteamOS, Bazzite and other immutable distributions). The libraries the game needs are
placed beside it. Only where a system allows no unprivileged sandboxes does the installer fall back
to the distribution's packages (pacman, apt, dnf or zypper, after asking for your password). On
Windows the build tools are MSYS2, set up automatically.

**Steam Deck (work in progress, see above):** run the AppImage in Desktop Mode. Afterwards open
Steam, choose *Games → Add a Non-Steam Game* and tick TOoT:R, and it appears in Gaming Mode. A first
start on a Deck picks the Steam Deck graphics preset.

From a clone of this repository the same installation runs in a terminal: `./install.sh` (Linux,
`--nup FILE`, `--dir FOLDER`, `--no-deps`) or `install.cmd` (Windows).

## Playing

Start **TOoT:R** from the menu, or run `tootr`. (Starting `app/tootr` in the install folder by hand
works too.)

**F1** or **Insert** shows or hides the settings window (**Select + Start** together on a
controller): graphics, game, cheats, hacks, controller, and keyboard and mouse, each on its own tab. Every control
can be rebound there: click a binding, then press the new key, mouse button or controller button.

- **Controller:** any SDL gamepad, with rumble and gyro aiming. Nintendo or Xbox button layout,
  dead zone, camera inversion.
- **Keyboard and mouse:** WASD move, the mouse turns the camera (Escape frees the mouse, a click
  takes it back), left click attack (Y), right click target (ZL), E = A, Shift = B, Space = X,
  F = ZR, Q/R = L/R, Enter = +, M = −, 1-4 the D-pad, Caps Lock walks. **F11** or Alt+Enter
  switches fullscreen.

Data (saves, settings, shader caches) lives in the install folder.

**If something goes wrong:** every run writes a log to `logs/tootr.log` in the install folder (the three
runs before it are kept beside it, with the launcher's and the last build's output). A fatal error
shows a message with the reason. Attach `tootr.log` when you report a problem.

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

The game is playable from start on PC with a controller or keyboard and mouse. Some rendering
details are still in progress, the Windows port is less tested than Linux, and the Steam Deck is a
work in progress. Please report issues.

## License

GPL-3.0-or-later (see `LICENSE`). Third-party code under `third_party/` keeps its own license and
notices; see the PROVENANCE/LICENSE files there.
