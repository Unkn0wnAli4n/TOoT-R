#!/usr/bin/env bash
# TOoT:R installer for Linux.
#
#   ./install.sh [--nup /path/to/game.nup] [--dir /install/folder] [--no-deps]
#
# Installs the build dependencies (asks for sudo), asks for your own dump of the game and for the
# folder to install into (default ~/.local/share/tootr), builds the game there, and installs the
# `tootr` launcher (~/.local/bin) plus an application menu entry. The launcher checks GitHub for
# a newer version on every start and rebuilds automatically (set OOT_RECOMP_NO_UPDATE=1 to skip).
# Run from a downloaded copy of the repository, or standalone: it then downloads the source itself.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
nup=""
dir=""
deps=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nup) nup="$2"; shift 2 ;;
        --dir) dir="$2"; shift 2 ;;
        --no-deps) deps=0; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

say() { printf '\033[1m[TOoT:R]\033[0m %s\n' "$*"; }

if [[ "$(uname -m)" != "x86_64" ]]; then
    say "only x86-64 PCs are supported (this is $(uname -m))"; exit 1
fi

install_deps() {
    local sudo=""
    [[ $EUID -ne 0 ]] && sudo="sudo"
    if command -v pacman >/dev/null; then
        $sudo pacman -S --needed --noconfirm clang lld cmake ninja python pkgconf sdl3 opus ffmpeg \
            vulkan-icd-loader vulkan-headers fmt zstd lz4 jemalloc curl
    elif command -v apt-get >/dev/null; then
        $sudo apt-get update
        $sudo apt-get install -y clang lld cmake ninja-build python3 pkg-config libsdl3-dev libopus-dev \
            libavcodec-dev libavutil-dev libswscale-dev libvulkan-dev libfmt-dev libzstd-dev liblz4-dev \
            libjemalloc-dev curl
    elif command -v dnf >/dev/null; then
        $sudo dnf install -y clang lld cmake ninja-build python3 pkgconf-pkg-config SDL3-devel opus-devel \
            ffmpeg-free-devel vulkan-loader-devel vulkan-headers fmt-devel libzstd-devel lz4-devel \
            jemalloc-devel curl
    elif command -v zypper >/dev/null; then
        $sudo zypper install -y clang lld cmake ninja python3 pkg-config SDL3-devel libopus-devel \
            ffmpeg-7-libavcodec-devel ffmpeg-7-libavutil-devel ffmpeg-7-libswscale-devel vulkan-devel \
            fmt-devel libzstd-devel liblz4-devel jemalloc-devel curl
    else
        say "unknown distribution: install clang, lld, cmake, ninja, python3, pkg-config and the"
        say "development files of SDL3, opus, FFmpeg, Vulkan, fmt, zstd and lz4, then run with --no-deps"
        exit 1
    fi
}

if [[ $deps -eq 1 ]]; then
    say "installing build dependencies (needs your password)"
    install_deps
fi

source_dir="$here"
if [[ ! -f "$here/tools/build_game.py" ]]; then
    slug="${OOT_RECOMP_REPO:-Unkn0wnAli4n/TOoT-R}"  # (keep in sync with installer/repo.txt)
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    say "downloading $slug"
    curl -fsSL "https://codeload.github.com/$slug/tar.gz/main" | tar -xz -C "$tmp" --strip-components=1
    source_dir="$tmp"
fi

args=(install --source "$source_dir")
[[ -n "$nup" ]] && args+=(--nup "$nup")
[[ -n "$dir" ]] && args+=(--dir "$dir")
python3 "$source_dir/installer/launcher.py" "${args[@]}"

case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) say "note: add ~/.local/bin to your PATH to start the game with 'tootr'" ;;
esac
