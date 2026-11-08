#!/usr/bin/env bash
# Builds the Windows installer, dist/TOoT-R-Installer.exe (installer/windows/setup.cpp with install.ps1
# built in), with clang for the x86_64-w64-mingw32 target. MINGW_SYSROOT: a MinGW-w64 sysroot (for example
# MSYS2's clang64 directory: include/ and lib/ with the Windows import libraries).
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
sysroot="${MINGW_SYSROOT:?set MINGW_SYSROOT to a MinGW-w64 sysroot (e.g. an MSYS2 clang64 directory)}"
work="$root/.dev/package/windows"; mkdir -p "$work" "$root/dist"
python3 - "$root/install.ps1" "$work/setup_script.h" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
with open(sys.argv[2], "w") as f:
    f.write("// Generated from install.ps1 by tools/package/windows.sh\n#pragma once\n")
    f.write("static const unsigned char kInstallScript[] = {\n")
    for i in range(0, len(data), 24):
        f.write(",".join(str(b) for b in data[i:i + 24]) + ",\n")
    f.write("};\nstatic const unsigned long long kInstallScriptSize = %d;\n" % len(data))
PY
# (the clang64 sysroot uses compiler-rt and libunwind, linked below: empty stand-ins for the GCC runtime the driver names)
for lib in gcc gcc_eh; do printf '!<arch>\n' > "$work/lib$lib.a"; done
python3 "$root/tools/package/make_res.py" "$root/installer/tootr.ico" "$work/setup.res"
clang++ --target=x86_64-w64-mingw32 --sysroot="$sysroot" -nostdinc++ -isystem "$sysroot/include/c++/v1" \
    -isystem "$sysroot/include" -L"$sysroot/lib" -L"$work" -nostdlib++ \
    -std=c++17 -O2 -municode -mwindows -static -fuse-ld=lld -rtlib=platform -I"$work" \
    "$root/installer/windows/setup.cpp" "$work/setup.res" -o "$root/dist/TOoT-R-Installer.exe" \
    -lc++ -lc++abi -lunwind "$(find "$sysroot/lib/clang" -name 'libclang_rt.builtins-x86_64.a' | head -1)" \
    -lcomctl32 -lcomdlg32 -lshell32 -lole32 -luser32 -lgdi32
ls -la "$root/dist/TOoT-R-Installer.exe"
