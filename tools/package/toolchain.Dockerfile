# The build environment the Linux installer carries (tools/package/toolchain.sh): compilers, CMake, Ninja, Python and the
# libraries the game links, as Arch Linux of 2025-07-01 had them (glibc 2.41, FFmpeg 7.1: what SteamOS 3.7 and later run).
FROM archlinux:latest
RUN echo 'Server=https://archive.archlinux.org/repos/2025/07/01/$repo/os/$arch' > /etc/pacman.d/mirrorlist && \
    sed -i 's/^SigLevel.*/SigLevel = Never/' /etc/pacman.conf && \
    pacman -Syyuu --noconfirm --overwrite '*' && \
    pacman -S --noconfirm --needed clang lld cmake ninja python pkgconf sdl3 opus ffmpeg vulkan-headers \
        vulkan-icd-loader fmt zstd lz4 git patchelf && \
    pacman -Scc --noconfirm
# For the self-contained installer: the sandbox the build runs in on PCs whose system cannot take packages (Steam Deck).
RUN pacman -S --noconfirm --needed bubblewrap jemalloc tk && pacman -Scc --noconfirm
