#!/usr/bin/env python3
"""Builds TOoT:R from the user's own copy of the game.

    build_game.py --nup <game.nup> --data <data-dir> --prefix <install-dir> [--work <cache-dir>] [-j N]

Steps (each skipped when its result is already current):
  1. build the offline tools (recompiler/: oot-extract, recomp)
  2. extract main/sdk/subsdk0 from the .nup into <data-dir>, and find this game version's places: every hook site
     and address the port uses is searched for by what the code around it looks like (tools/signatures.py,
     data/signatures.txt), so a game update other than the tested one builds too, without what it lacks
  3. generate C++ from the images and compile it into a game_code archive. The archive is cached
     under <work> by a hash of everything the generated code depends on (the game's code, the places found, the
     recompiler, its data, runtime/ops.h, cmake/GameCode.cmake, compiler version); the ~2 GB of generated
     sources and the object files are deleted afterwards
  4. build the runtime against that archive and install the stripped executable into <prefix>

Works on Linux and on Windows inside MSYS2 (CLANG64). Only Python's standard library is used.
"""
import argparse, glob, hashlib, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = [  # name, load base, recompiler output file count
    ("main", "800000000", 512),
    ("subsdk0", "810000000", 8),
    ("sdk", "820000000", 96),
]
WINDOWS = os.name == "nt" or sys.platform == "msys" or sys.platform.startswith("cygwin")
EXE = ".exe" if WINDOWS else ""


def log(msg):
    print(f"[build] {msg}", flush=True)


def step(key):
    """A line the graphical installers read (installer/gui.py, installer/windows/setup.cpp): which part begins."""
    print(f"@step {key}", flush=True)


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


def game_key(data):
    """Hash of every input of the generated game code: the tools, and this game's own code and places."""
    h = hashlib.sha256()
    files = []
    for pattern in ["recompiler/*.cpp", "recompiler/CMakeLists.txt", "third_party/dynarmic/**/*",
                    "data/names/*.names", "tools/module_info.py", "tools/setjmp_targets.py",
                    "tools/oot_patches.py", "runtime/ops.h", "cmake/GameCode.cmake"]:
        files += [p for p in glob.glob(os.path.join(ROOT, pattern), recursive=True) if os.path.isfile(p)]
    for p in sorted(files):
        h.update(os.path.relpath(p, ROOT).replace("\\", "/").encode())
        with open(p, "rb") as f:
            h.update(f.read())
    with open(os.path.join(data, "images.sha256")) as f:  # (the code's own hashes: not which file it came from)
        h.update("".join(sorted(l for l in f if not l.rstrip().endswith(" source"))).encode())
    with open(os.path.join(data, "addresses.txt"), "rb") as f:
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


def known_versions():
    """{(main, sdk, subsdk0 hashes): name} of the game versions this project was tested with (data/v*/images.sha256)."""
    known = {}
    for path in glob.glob(os.path.join(ROOT, "data", "v*", "images.sha256")):
        digests = {}
        with open(path) as f:
            for line in f:
                digest, name = line.split()
                digests[name] = digest
        known[tuple(digests.get(n + ".img") for n, _, _ in MODULES)] = os.path.basename(os.path.dirname(path))
    return known


def exports(image):
    """{name: offset} of a module image's exported symbols (its dynamic symbol table)."""
    import struct
    with open(image, "rb") as f:
        img = f.read()
    mod0 = struct.unpack_from("<I", img, 4)[0]
    d = mod0 + struct.unpack_from("<i", img, mod0 + 4)[0]
    tags = {}
    while True:
        tag, val = struct.unpack_from("<qQ", img, d)
        if tag == 0:
            break
        tags.setdefault(tag, val)
        d += 16
    symtab, strtab, out, i = tags[6], tags[5], {}, 1
    while symtab + (i + 1) * 24 <= strtab:
        name, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", img, symtab + i * 24)
        if value and shndx:
            out.setdefault(img[strtab + name:img.index(b"\0", strtab + name)].decode(), value)
        i += 1
    return out


def extract(nup, tools, data):
    """The game's code out of the .nup, and what this version of it is: its places (addresses.txt) and whether it is
    a version this project was tested with."""
    images = [n + ".img" for n, _, _ in MODULES]
    stamp_path = os.path.join(data, "images.sha256")
    # (which game file the images were taken from: another file, or the same one changed, is extracted again)
    source = ""
    if nup and os.path.isfile(nup):
        st = os.stat(nup)
        source = ("%s %d %d" % (os.path.basename(nup), st.st_size, int(st.st_mtime))).replace(" ", "_")
    recorded = {}
    if os.path.exists(stamp_path):
        with open(stamp_path) as f:
            for line in f:
                p = line.rstrip("\n").split(None, 1)
                if len(p) == 2:
                    recorded[p[1]] = p[0]
    have = all(os.path.exists(os.path.join(data, n)) and os.path.exists(os.path.join(data, n + ".text-size")) for n in images)
    known = known_versions()
    if have and not recorded:  # (an installation from before other versions were taken: its images are a tested version's or are made again)
        digests = tuple(sha256_file(os.path.join(data, n)) for n in images)
        if digests in known:
            recorded = dict(zip(images, digests))
            recorded["source"] = source
    current = have and all(n in recorded for n in images) and (not source or recorded.get("source", "-") in ("-", source))
    if not current:
        if not nup or not os.path.isfile(nup):
            fail("the game file (.nup) was not found: " + str(nup))
        log("extracting the game's code from " + nup)
        for name in images:  # (older setups linked these to a development tree)
            if os.path.islink(os.path.join(data, name)):
                os.remove(os.path.join(data, name))
        run([os.path.join(tools, "oot-extract" + EXE), nup, data])
        recorded = {n: sha256_file(os.path.join(data, n)) for n in images}
    recorded["source"] = source or recorded.get("source", "")
    with open(stamp_path, "w") as f:
        for n in images:
            f.write("%s %s\n" % (recorded[n], n))
        f.write("%s source\n" % (recorded["source"] or "-"))
    digests = tuple(recorded[n] for n in images)
    version = known.get(digests)
    if version:
        log(f"game code: update {version}, a version this build was tested with")
    else:
        log("game code: NOT a version this build was tested with (tested: %s). It is built from what is found in it; "
            "what this version lacks is left out, and it may not play as the tested version does." % ", ".join(sorted(known.values())))
    # The places of this game version.
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import signatures
    found, lost = signatures.find(data, os.path.join(ROOT, "data", "signatures.txt"), os.path.join(data, "addresses.txt"), quiet=True)
    moved = sum(1 for (m, o), v in found.items() if o != v)
    log("game code: %d of %d places found%s" % (len(found), len(found) + len(lost), " (%d moved from the tested version's)" % moved if moved else ""))
    for module, off in lost:
        log("  not in this game version: %s+%#x (what uses it is left out)" % (module, off))
    if version and lost:
        fail("the tested version's own places were not all found: tools/signatures.py or data/signatures.txt is out of date")
    # The graphics driver's function map: the tested SDK's is shipped; another SDK's is made by the game's first start.
    names = os.path.join(ROOT, "data", "names", "nvn.names")
    shutil.copyfile(names, os.path.join(data, "nvn.names"))
    sdk_map = None
    for digests_known, name in known.items():
        if dict(zip(images, digests_known)).get("sdk.img") == recorded["sdk.img"]:
            sdk_map = os.path.join(ROOT, "data", name, "nvn_map.txt")
    target = os.path.join(data, "nvn_map.txt")
    if sdk_map:
        shutil.copyfile(sdk_map, target)
    else:
        if os.path.exists(target):
            os.remove(target)
        log("game code: this version's SDK is not the tested one: its graphics functions are mapped when the game first starts")


def generate(tools, data, work):
    """C++ for every module into <work>/gen (inputs copied next to the images in <work>/meta)."""
    meta = os.path.join(work, "meta")
    gen = os.path.join(work, "gen")
    shutil.rmtree(gen, ignore_errors=True)
    os.makedirs(meta, exist_ok=True)
    for name, _, _ in MODULES:
        for suffix in [".img", ".img.text-size"]:
            shutil.copyfile(os.path.join(data, name + suffix), os.path.join(meta, name + suffix))
        # (the SDK functions the runtime implements itself, by name: where each is in this version's module)
        have = exports(os.path.join(meta, name + ".img"))
        with open(os.path.join(ROOT, "data", "names", name + ".names")) as f:
            wanted = [l.strip() for l in f if l.strip()]
        with open(os.path.join(meta, name + ".overrides"), "w") as f:
            f.write("".join(sorted("%x %s\n" % (have[n], n) for n in wanted if n in have)))
        absent = [n for n in wanted if n not in have]
        if absent:
            log("  %s: %d of the functions the runtime replaces are not in this version (%s%s)" % (name, len(absent), ", ".join(absent[:3]), ", ..." if len(absent) > 3 else ""))
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
        entries = os.path.join(meta, name + ".entries")  # (extra block starts: the hook sites of tools/oot_patches.py, where they are in this game)
        if name == "main":
            import oot_patches
            with open(entries, "w") as f:
                f.write(" ".join("%x" % a for a in oot_patches.block_starts(os.path.join(data, "addresses.txt"))) + "\n")
        if os.path.exists(entries):
            cmd += ["--entries", entries]
        setjmp = os.path.join(meta, name + ".setjmp")
        if os.path.getsize(setjmp):
            cmd += ["--setjmp", setjmp]
        log(f"recompiling {name}")
        with open(os.path.join(work, name + ".recomp.log"), "w") as errors:
            run(cmd, stderr=errors)
    run([sys.executable, "-P", os.path.join(ROOT, "tools", "oot_patches.py"), gen, os.path.join(data, "addresses.txt")])
    shutil.rmtree(meta, ignore_errors=True)
    return gen


def build_game_code(tools, data, work, jobs):
    key = game_key(data)
    lib_name = "libgame_code.a"
    cached = os.path.join(work, f"game_code-{key}.a")
    if os.path.exists(cached):
        log(f"game code is up to date ({key})")
        return cached
    log("generating and compiling the game code (the long step: 15-60 minutes)")
    start = time.time()
    step("translate")
    gen = generate(tools, data, work)
    build = os.path.join(work, "game-build")
    shutil.rmtree(build, ignore_errors=True)
    cmake_configure(ROOT, build, [f"-DGEN_DIR={gen}", "-DOOT_GAME_ONLY=ON"])
    step("compile")
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
    # (NVIDIA DLSS is built in only with NVIDIA's SDK, which is never part of this project: OOT_NGX_SDK=<its folder>)
    ngx = os.environ.get("OOT_NGX_SDK", "")
    cmake_configure(ROOT, build, [f"-DOOT_GAME_LIB={game_lib}", f"-DCMAKE_INSTALL_PREFIX={prefix}",
                                  f"-DOOT_DEBUG_INFO={'ON' if debug else 'OFF'}", f"-DOOT_NGX_SDK={ngx}"])
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
    step("tools")
    tools = build_tools(work, jobs)
    step("check")
    extract(a.nup and os.path.abspath(a.nup), tools, data)
    game_lib = build_game_code(tools, data, work, jobs)
    step("runtime")
    build_runtime(game_lib, work, prefix, jobs, a.debug)
    log("done: " + os.path.join(prefix, "tootr" + EXE))


if __name__ == "__main__":
    main()
