#!/usr/bin/env bash
# Builds the Linux installer: dist/TOoT-R-Installer-x86_64.AppImage (installer/gui.py with a bundled
# Python and Tk, this source tree, and the build environment of tools/package/toolchain.sh, so that the
# installation needs nothing from the system: a Steam Deck's is read-only; nothing from the game).
# Needs curl, network access and docker (for the build environment; NO_TOOLCHAIN=1 leaves it out: the
# installer then uses the system's packages).
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$root/.dev/package"; app="$work/AppDir"; dist="$root/dist"
tool_url="https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"
runtime_url="https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64"
mkdir -p "$work" "$dist"
fetch() { [[ -s "$2" ]] || { curl -fL --retry 8 --retry-delay 5 --retry-all-errors -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
fetch "$tool_url" "$work/appimagetool"
fetch "$runtime_url" "$work/runtime-x86_64"
chmod +x "$work/appimagetool"
[[ -s "$work/toolchain.tar.xz" && -d "$work/toolchain/usr/lib/tk8.6" ]] || "$root/tools/package/toolchain.sh"
tree="$work/toolchain"

# The installer's own Python and Tk come from the build environment too (with its C library and loader, so the window
# starts on any distribution, and Tk draws with the system's fonts): the interpreter, its library, Tcl/Tk, and every
# shared library those load.
rm -rf "$app"; mkdir -p "$app/usr/bin" "$app/usr/lib" "$app/source"
py="$(basename "$(ls -d "$tree"/usr/lib/python3.* | head -1)")"
cp "$tree/usr/bin/$py" "$app/usr/bin/python3"
cp -a "$tree/usr/lib/$py" "$app/usr/lib/$py"
rm -rf "$app/usr/lib/$py/"{test,idlelib,ensurepip,turtledemo,lib2to3,pydoc_data,site-packages,config-*}
cp -a "$tree"/usr/lib/tcl8.6 "$tree"/usr/lib/tk8.6 "$app/usr/lib/"
docker run --rm tootr-toolchain sh -c "ldd /usr/bin/$py /usr/lib/$py/lib-dynload/*.so /usr/lib/libtk8.6.so 2>/dev/null | awk '/=> \\//{print \$3}' | sort -u" |
    while read -r lib; do cp -L "$tree$lib" "$app/usr/lib/" 2>/dev/null || true; done
cp -L "$tree/usr/lib/ld-linux-x86-64.so.2" "$app/usr/lib/"
cat > "$app/usr/bin/tootr-python" <<'PY'
#!/bin/sh
# The installer's Python, on its own C library (any distribution runs it).
usr="$(dirname "$(dirname "$(readlink -f "$0")")")"
PYTHONHOME="$usr" TCL_LIBRARY="$usr/lib/tcl8.6" TK_LIBRARY="$usr/lib/tk8.6" exec "$usr/lib/ld-linux-x86-64.so.2" --library-path "$usr/lib" "$usr/bin/python3" "$@"
PY
chmod +x "$app/usr/bin/tootr-python"
# The source tree as committed (no build outputs or development files).
if git -C "$root" rev-parse --git-dir > /dev/null 2>&1; then
    ( cd "$root" && git ls-files -z ) | ( cd "$root" && xargs -0 cp --parents -t "$app/source" )
else  # (not a repository yet: every file and folder but the development ones)
    ( cd "$root" && find . -mindepth 1 \( -path ./.dev -o -path ./dist -o -path ./.git -o -name __pycache__ \) -prune -o -type f -print0 ) |
        ( cd "$root" && xargs -0 cp --parents -t "$app/source" )
fi
if [[ -z "${NO_TOOLCHAIN:-}" ]]; then
    ln -f "$work/toolchain.tar.xz" "$app/toolchain.tar.xz" 2>/dev/null || cp "$work/toolchain.tar.xz" "$app/toolchain.tar.xz"
fi
cat > "$app/AppRun" <<'RUN'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
export TOOTR_SOURCE="$here/source" TOOTR_PYTHON="$here/usr/bin/tootr-python"
[ -f "$here/toolchain.tar.xz" ] && export TOOTR_TOOLCHAIN="$here/toolchain.tar.xz"
exec "$here/usr/bin/tootr-python" "$here/source/installer/gui.py" "$@"
RUN
chmod +x "$app/AppRun"
cat > "$app/tootr-installer.desktop" <<'DESK'
[Desktop Entry]
Type=Application
Name=TOoT:R Installer
Comment=Builds TOoT:R from your own copy of the game
Exec=AppRun
Icon=tootr
Categories=Game;
Terminal=false
DESK
cp "$root/installer/tootr.png" "$app/tootr.png"
ln -sf tootr.png "$app/.DirIcon"
ARCH=x86_64 "$work/appimagetool" --appimage-extract-and-run --no-appstream --runtime-file "$work/runtime-x86_64" "$app" "$dist/TOoT-R-Installer-x86_64.AppImage"
ls -la "$dist/TOoT-R-Installer-x86_64.AppImage"
