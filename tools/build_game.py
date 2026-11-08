#!/usr/bin/env python3
"""Builds TOoT:R from the user's own copy of the game.

    build_game.py --nup <game.nup> --data <data-dir> --prefix <install-dir> [--work <cache-dir>] [-j N]

Steps (each skipped when its result is already current):
  1. build the offline tools (recompiler/: oot-extract, recomp)
  2. extract main/sdk/subsdk0 from the .nup into <data-dir> and check them against data/<version>
  3. generate C++ from the images and compile it into a game_code archive. The archive is cached
     under <work> by a hash of everything the generated code depends on (recompiler, its data,
     runtime/ops.h, cmake/GameCode.cmake, compiler version); the ~2 GB of generated sources and the
     object files are deleted afterwards
  4. build the runtime against that archive and install the stripped executable into <prefix>

Works on Linux and on Windows inside MSYS2 (CLANG64). Only Python's standard library is used.
"""
import argparse, glob, hashlib, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "v131072"
MODULES = [  # name, load base, recompiler output file count
    ("main", "800000000", 512),
    ("subsdk0", "810000000", 8),
    ("sdk", "820000000", 96),
]
WINDOWS = os.name == "nt" or sys.platform == "msys" or sys.platform.startswith("cygwin")
EXE = ".exe" if WINDOWS else ""


def log(msg):
    print(f"[build] {msg}", flush=True)


def fail(msg):
    print(f"[build] error: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(cmd, **kw):
    log(" ".join(str(c) for c in cmd) if len(cmd) < 12 else f"{cmd[0]} ...")
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        fail(f"command failed ({r.returncode}): {cmd[0]}")
    return r


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_jobs():
    cpus = os.cpu_count() or 4
    mem_gb = 0
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "ave")]
            m = MS(); m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            mem_gb = m.tp / 2**30
        except Exception:
            pass
    # The largest generated files need ~1.5 GB of compiler memory each.
    by_mem = int(mem_gb / 1.5) if mem_gb else cpus
    return max(1, min(cpus, by_mem))


TOOLCHAIN = []  # (--toolchain: cross builds of the game and runtime; the offline tools stay native)


def cmake_configure(src, build, extra, target=True):
    cmd = ["cmake", "-S", src, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
           "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"] + (TOOLCHAIN if target else []) + extra
    run(cmd)


def compiler_id():
    try:
        return subprocess.run(["clang++", "--version"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        fail("clang++ not found (the installer normally sets it up)")


def game_key():
    """Hash of every input of the generated game code (not the game itself: that is checked by hash)."""
    h = hashlib.sha256()
    files = []
    for pattern in ["recompiler/*.cpp", "recompiler/CMakeLists.txt", "third_party/dynarmic/**/*",
                    f"data/{VERSION}/*.overrides", "tools/module_info.py", "tools/setjmp_targets.py",
                    "tools/oot_patches.py", "runtime/ops.h", "cmake/GameCode.cmake"]:
        files += [p for p in glob.glob(os.path.join(ROOT, pattern), recursive=True) if os.path.isfile(p)]
    for p in sorted(files):
        h.update(os.path.relpath(p, ROOT).replace("\\", "/").encode())
        with open(p, "rb") as f:
            h.update(f.read())
    h.update(compiler_id().encode())
    h.update(" ".join(TOOLCHAIN).encode())
    return h.hexdigest()[:16]


def build_tools(work, jobs):
    tools = os.path.join(work, "tools")
    if not os.path.exists(os.path.join(tools, "build.ninja")):
        cmake_configure(os.path.join(ROOT, "recompiler"), tools, [], target=False)
    run(["cmake", "--build", tools, "-j", str(jobs)])
    return tools


def extract(nup, tools, data):
    expected = {}
    with open(os.path.join(ROOT, "data", VERSION, "images.sha256")) as f:
        for line in f:
            digest, name = line.split()
            expected[name] = digest
    current = all(os.path.exists(os.path.join(data, n)) and os.path.exists(os.path.join(data, n + ".text-size"))
                  and sha256_file(os.path.join(data, n)) == d for n, d in expected.items())
    if not current:
        if not nup or not os.path.isfile(nup):
            fail("the game file (.nup) was not found: " + str(nup))
        log("extracting the game's code from " + nup)
        for name in expected:  # (older setups linked these to a development tree)
            if os.path.islink(os.path.join(data, name)):
                os.remove(os.path.join(data, name))
        run([os.path.join(tools, "oot-extract" + EXE), nup, data])
        for name, digest in expected.items():
            if sha256_file(os.path.join(data, name)) != digest:
                fail(f"{name} does not match the supported version ({VERSION}). This build needs "
                     "The Legend of Zelda: Ocarina of Time for Nintendo Switch 2 with update "
                     f"{VERSION} (title 0400486024DEC000), dumped as a single .nup.")
    shutil.copyfile(os.path.join(ROOT, "data", VERSION, "nvn_map.txt"), os.path.join(data, "nvn_map.txt"))
    log("game code verified")


def generate(tools, data, work):
    """C++ for every module into <work>/gen (inputs copied next to the images in <work>/meta)."""
    meta = os.path.join(work, "meta")
    gen = os.path.join(work, "gen")
    shutil.rmtree(gen, ignore_errors=True)
    os.makedirs(meta, exist_ok=True)
    for name, _, _ in MODULES:
        for suffix in [".img", ".img.text-size"]:
            shutil.copyfile(os.path.join(data, name + suffix), os.path.join(meta, name + suffix))
        shutil.copyfile(os.path.join(ROOT, "data", VERSION, name + ".overrides"), os.path.join(meta, name + ".overrides"))
        run([sys.executable, "-I", os.path.join(ROOT, "tools", "module_info.py"),
             os.path.join(meta, name + ".img"), os.path.join(meta, name + ".functions.txt")])
    run([sys.executable, "-I", os.path.join(ROOT, "tools", "setjmp_targets.py"), meta])
    for name, base, count in MODULES:
        out = os.path.join(gen, name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(meta, name + ".img.text-size")) as f:
            text_size = f.read().strip()
        cmd = [os.path.join(tools, "recomp" + EXE), os.path.join(meta, name + ".img"), base,
               os.path.join(meta, name + ".functions.txt"), out, name,
               "--overrides", os.path.join(meta, name + ".overrides"), "--files", count, "--text-size", text_size]
        setjmp = os.path.join(meta, name + ".setjmp")
        if os.path.getsize(setjmp):
            cmd += ["--setjmp", setjmp]
        log(f"recompiling {name}")
        with open(os.path.join(work, name + ".recomp.log"), "w") as errors:
            run(cmd, stderr=errors)
    run([sys.executable, "-P", os.path.join(ROOT, "tools", "oot_patches.py"), gen])
    shutil.rmtree(meta, ignore_errors=True)
    return gen


def build_game_code(tools, data, work, jobs):
    key = game_key()
    lib_name = "libgame_code.a"
    cached = os.path.join(work, f"game_code-{key}.a")
    if os.path.exists(cached):
        log(f"game code is up to date ({key})")
        return cached
    log("generating and compiling the game code (the long step: 15-60 minutes)")
    start = time.time()
    gen = generate(tools, data, work)
    build = os.path.join(work, "game-build")
    shutil.rmtree(build, ignore_errors=True)
    cmake_configure(ROOT, build, [f"-DGEN_DIR={gen}", "-DOOT_GAME_ONLY=ON"])
    run(["cmake", "--build", build, "--target", "game_code", "-j", str(jobs)])
    for old in glob.glob(os.path.join(work, "game_code-*.a")):
        os.remove(old)
    shutil.move(os.path.join(build, lib_name), cached)
    shutil.rmtree(build, ignore_errors=True)
    shutil.rmtree(gen, ignore_errors=True)
    log(f"game code built in {int(time.time() - start) // 60} min")
    return cached


def build_runtime(game_lib, work, prefix, jobs, debug):
    build = os.path.join(work, "runtime-build")
    cmake_configure(ROOT, build, [f"-DOOT_GAME_LIB={game_lib}", f"-DCMAKE_INSTALL_PREFIX={prefix}",
                                  f"-DOOT_DEBUG_INFO={'ON' if debug else 'OFF'}"])
    run(["cmake", "--build", build, "-j", str(jobs)])
    run(["cmake", "--install", build] + ([] if debug else ["--strip"]))
    if not debug:  # (the installed copy is the one that runs; objects stay for incremental builds)
        for name in ("tootr", "tootr.exe"):
            if os.path.exists(os.path.join(build, name)):
                os.remove(os.path.join(build, name))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nup", help="the game (.nup); only needed until the code has been extracted")
    ap.add_argument("--data", required=True, help="data directory (module images, saves, caches)")
    ap.add_argument("--prefix", required=True, help="where to install the executable")
    ap.add_argument("--work", help="build cache (default: <data>/build)")
    ap.add_argument("-j", "--jobs", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="keep debug info in the runtime (development)")
    ap.add_argument("--toolchain", help="CMake toolchain file for the game and runtime (cross builds)")
    a = ap.parse_args()
    data = os.path.abspath(a.data)
    work = os.path.abspath(a.work or os.path.join(data, "build"))
    prefix = os.path.abspath(a.prefix)
    jobs = a.jobs or default_jobs()
    if a.toolchain:
        TOOLCHAIN.append("-DCMAKE_TOOLCHAIN_FILE=" + os.path.abspath(a.toolchain))
    for d in (data, work, prefix):
        os.makedirs(d, exist_ok=True)
    for tool in ("cmake", "ninja", "clang++"):
        if not shutil.which(tool):
            fail(f"{tool} is not installed")
    tools = build_tools(work, jobs)
    extract(a.nup and os.path.abspath(a.nup), tools, data)
    game_lib = build_game_code(tools, data, work, jobs)
    build_runtime(game_lib, work, prefix, jobs, a.debug)
    log("done: " + os.path.join(prefix, "tootr" + EXE))


if __name__ == "__main__":
    main()
