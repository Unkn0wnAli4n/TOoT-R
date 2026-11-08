#!/usr/bin/env bash
# Builds the Linux installer: dist/TOoT-R-Installer-x86_64.AppImage (installer/gui.py with a bundled
# Python and Tk, and this source tree; nothing from the game). Needs curl and network access.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$root/.dev/package"; app="$work/AppDir"; dist="$root/dist"
python_url="https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.12.15+20261013-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
tool_url="https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"
runtime_url="https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64"
mkdir -p "$work" "$dist"
fetch() { [[ -s "$2" ]] || { curl -fL --retry 8 --retry-delay 5 --retry-all-errors -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
# (Tk 9: the Tk 8.6 of earlier builds aborted in Xlib as soon as a text entry was shown)
fetch "$python_url" "$work/python-20261013.tar.gz"
fetch "$tool_url" "$work/appimagetool"
fetch "$runtime_url" "$work/runtime-x86_64"
chmod +x "$work/appimagetool"

rm -rf "$app"; mkdir -p "$app/usr" "$app/source"
tar -xzf "$work/python-20261013.tar.gz" -C "$app/usr"   # -> usr/python
rm -rf "$app/usr/python/lib/python3.12/"{test,idlelib,ensurepip,turtledemo,lib2to3,pydoc_data} \
       "$app/usr/python/include" "$app/usr/python/share"
# The source tree as committed (no build outputs or development files).
( cd "$root" && git ls-files -z ) | ( cd "$root" && xargs -0 cp --parents -t "$app/source" )
cat > "$app/AppRun" <<'RUN'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
export TOOTR_SOURCE="$here/source"
exec "$here/usr/python/bin/python3" "$here/source/installer/gui.py" "$@"
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
