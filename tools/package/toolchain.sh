#!/usr/bin/env bash
# Makes the build environment the Linux installer carries: .dev/package/toolchain.tar.xz, the file system of the
# tools/package/toolchain.Dockerfile image without what a build never reads (documentation, translations, package
# databases, static libraries nobody links, the Python test suite). Needs docker. The installer unpacks it into
# <install folder>/toolchain and runs the build inside it (installer/launcher.py), so nothing is installed on the PC.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$root/.dev/package"; tree="$work/toolchain"; mkdir -p "$work"
docker build -q -t tootr-toolchain -f "$root/tools/package/toolchain.Dockerfile" "$root/tools/package" > /dev/null
[[ -d "$tree" ]] && { chmod -R u+w "$tree"; rm -rf "$tree"; }; mkdir -p "$tree"
id=$(docker create tootr-toolchain true)
docker export "$id" | tar -x -C "$tree" --anchored --exclude='dev/*' --exclude='proc/*' --exclude='sys/*' --no-same-owner 2>/dev/null || true
docker rm "$id" > /dev/null
chmod -R u+rwX "$tree"
cd "$tree"
rm -rf usr/share/{doc,man,info,locale,gtk-doc,i18n,zoneinfo,licenses,fonts,icons,terminfo/[!xl]*,vim,perl5,gir-1.0,bash-completion,zsh,fish,qt6,X11,opencv4,gdb,gcc-*/python} \
       var/cache var/lib/pacman var/log usr/lib/python3*/test usr/lib/python3*/{idlelib,turtledemo,ensurepip} \
       usr/lib/{systemd,udev,modules,firmware,gconv,girepository-1.0,perl5,ruby,security,kernel,sysusers.d,tmpfiles.d,bfd-plugins/../../share/locale} \
       usr/include/{qt6,boost,opencv4} etc/{pacman.d,ssl/certs/java} boot home mnt opt srv root run media
find usr/lib -name '*.a' ! -name 'libc_nonshared.a' ! -name 'libgcc*.a' ! -name 'libclang_rt*' ! -name 'libstdc++*.a' ! -name 'libsupc++.a' ! -name 'libm*.a' ! -name 'libpthread.a' ! -name 'libdl.a' ! -name 'librt.a' ! -name 'libutil.a' ! -name 'libg.a' ! -name 'libatomic.a' -delete
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
# (programs a build never starts; the libraries stay, the game links some of them)
( cd usr/bin && ls | grep -E '^(git|perl|gdb|systemd|journalctl|pacman|gpg|openssl|ffmpeg|ffprobe|ffplay|qt|x86_64-pc-linux-gnu-g|gcc|g\+\+|cpp|lto-dump|gcov|gfortran|c\+\+$|cc$)' | xargs -r rm -f ) || true
rm -rf usr/lib/gcc/x86_64-pc-linux-gnu/*/{cc1,cc1plus,lto1,lto-wrapper,collect2,plugin} usr/lib/git-core usr/share/git* usr/lib/libgfortran* usr/lib/libgo.* usr/lib/libgphobos* usr/lib/libgdruntime*
mkdir -p tmp dev proc sys run var/tmp
du -sh "$tree" | sed 's/^/toolchain tree: /'
cd "$work"
tar -C "$tree" -cf - . | xz -T0 -6 > "$work/toolchain.tar.xz"
ls -la "$work/toolchain.tar.xz"
