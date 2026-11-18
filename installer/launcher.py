#!/usr/bin/env python3
"""TOoT:R launcher, installer back end and auto-updater (Linux and Windows/MSYS2).

    launcher.py install [--nup PATH] [--dir DIR] [--source DIR] [--toolchain ARCHIVE]
                                                       first install (asks for the game file and where
                                                       to install; the default place when none is given)
    launcher.py run [game args...]                     update if GitHub has a newer version, then play
    launcher.py update [--force]                       only update
    launcher.py set-game [PATH]                        another copy of the game for this installation (asks when no
                                                       path is given; "tootr --game [PATH]" does the same, then plays)

Layout under the install directory (by default ~/.local/share/tootr, or %LOCALAPPDATA%\\tootr; the
launcher finds the one it was installed into, OOT_RECOMP_HOME overrides):
    source/     this project's source (replaced on update)
    app/        the built executable
    build/      build cache (offline tools, the compiled game code archive, runtime build)
    toolchain/  the build environment the installer brought (Linux, optional: else the system's tools)
    config.json game path, installed commit
    save/, shader_cache/, pipeline_cache.bin, graphics.ini   created by the game
The game's own files never leave the user's machine and are never uploaded anywhere.
"""
import json, os, shutil, subprocess, sys, tarfile, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = os.name == "nt" or sys.platform == "msys" or sys.platform.startswith("cygwin")
EXE = ".exe" if WINDOWS else ""
BRANCH = "main"
CHECK_TIMEOUT = 5


def default_dir():
    if WINDOWS:
        return os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "tootr")
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "tootr")


def installed_dir():
    """The install directory: OOT_RECOMP_HOME, else the one this launcher runs from (<dir>/source/installer),
    else the default."""
    if os.environ.get("OOT_RECOMP_HOME"):
        return os.environ["OOT_RECOMP_HOME"]
    here_root = os.path.dirname(os.path.dirname(HERE))
    if os.path.basename(os.path.dirname(HERE)) == "source" and os.path.exists(os.path.join(here_root, "config.json")):
        return here_root
    return default_dir()


def set_root(path):
    global ROOT, SOURCE, APP, CONFIG
    ROOT = os.path.abspath(os.path.expanduser(path))
    SOURCE = os.path.join(ROOT, "source")
    APP = os.path.join(ROOT, "app")
    CONFIG = os.path.join(ROOT, "config.json")


set_root(installed_dir())


def log_dir():
    """<install folder>/logs: the game's own log of each run (tootr.log, the three before it), the launcher's
    (launcher.log) and the last build's output (build.log). What a problem report needs."""
    path = os.path.join(ROOT, "logs")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    return path


def say(msg):
    print(f"[TOoT:R] {msg}", flush=True)
    try:  # (kept, with the time: a console window that closed took its lines with it)
        logs = log_dir()
        if logs:
            path = os.path.join(logs, "launcher.log")
            if os.path.exists(path) and os.path.getsize(path) > (1 << 20):
                os.replace(path, path + ".old")
            with open(path, "a", encoding="utf-8", errors="replace") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + str(msg) + "\n")
    except OSError:
        pass


def run_logged(cmd, name):
    """cmd run with its output shown as it comes and kept in logs/<name>.log (the one before it as <name>.1.log)."""
    logs = log_dir()
    if not logs:
        return subprocess.run(cmd).returncode
    path = os.path.join(logs, name + ".log")
    try:
        if os.path.exists(path):
            os.replace(path, os.path.join(logs, name + ".1.log"))
        out = open(path, "wb")
    except OSError:
        return subprocess.run(cmd).returncode
    with out:
        out.write((time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(c) for c in cmd) + "\n").encode("utf-8", "replace"))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        target = getattr(sys.stdout, "buffer", None)
        for line in iter(p.stdout.readline, b""):
            out.write(line); out.flush()
            try:
                if target:
                    target.write(line); target.flush()
                else:
                    sys.stdout.write(line.decode("utf-8", "replace")); sys.stdout.flush()
            except (OSError, ValueError):
                pass
        return p.wait()


def notify(text):
    """A message the player sees even without a console: a system dialog where one is at hand."""
    say(text)
    try:
        if WINDOWS:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, "TOoT:R", 0x10 | 0x10000)
        elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            for cmd in (["zenity", "--error", "--title", "TOoT:R", "--no-wrap", "--text", text], ["kdialog", "--title", "TOoT:R", "--error", text]):
                if shutil.which(cmd[0]):
                    subprocess.run(cmd, timeout=600)
                    break
    except Exception:
        pass


def load_config():
    try:
        with open(CONFIG) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    os.makedirs(ROOT, exist_ok=True)
    tmp = CONFIG + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG)


def repo_slug(source=None):
    """owner/name of the GitHub project, from installer/repo.txt."""
    for d in [source and os.path.join(source, "installer"), HERE]:
        if d and os.path.exists(os.path.join(d, "repo.txt")):
            with open(os.path.join(d, "repo.txt")) as f:
                slug = f.read().strip()
            if slug and "/" in slug and not slug.startswith("OWNER/"):
                return slug
    return None


def notify(message):
    """Desktop notification (the launcher usually runs without a terminal on Linux)."""
    if not WINDOWS and shutil.which("notify-send"):
        subprocess.run(["notify-send", "-a", "tootr", "TOoT:R", message], check=False)


# ---- GitHub ------------------------------------------------------------------------------------
def http_get(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "tootr-updater",
                                               "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(req, timeout=timeout)


def latest_commit(slug, timeout=CHECK_TIMEOUT):
    with http_get(f"https://api.github.com/repos/{slug}/commits/{BRANCH}", timeout) as r:
        return json.load(r)["sha"]


def download_source(slug, sha, dest):
    """Downloads and unpacks the source of one commit into dest (replaced atomically)."""
    say(f"downloading {slug}@{sha[:10]}")
    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        archive = os.path.join(tmp, "src.tar.gz")
        with http_get(f"https://codeload.github.com/{slug}/tar.gz/{sha}", 60) as r, open(archive, "wb") as out:
            shutil.copyfileobj(r, out)
        unpack = os.path.join(tmp, "unpacked")
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers() if not (m.issym() or m.islnk() or m.isdev())]
            for m in members:
                if m.name.startswith("/") or ".." in m.name.split("/"):
                    raise RuntimeError("unsafe path in source archive: " + m.name)
            try:
                t.extractall(unpack, members=members, filter="data")
            except TypeError:  # (Python without extraction filters)
                t.extractall(unpack, members=members)
        (top,) = os.listdir(unpack)
        new = os.path.join(unpack, top)
        stamp_changes(new, dest)
        replace_dir(new, dest)


def stamp_changes(new, old):
    """Archive files carry commit times, which can be older than the last build's outputs: give
    changed files the current time (so the build sees them) and unchanged ones their old time."""
    now = time.time()
    for base, _, files in os.walk(new):
        for name in files:
            path = os.path.join(base, name)
            previous = os.path.join(old, os.path.relpath(path, new))
            try:
                if os.path.getsize(path) == os.path.getsize(previous):
                    with open(path, "rb") as a, open(previous, "rb") as b:
                        if a.read() == b.read():
                            st = os.stat(previous)
                            os.utime(path, (st.st_atime, st.st_mtime))
                            continue
            except OSError:
                pass
            os.utime(path, (now, now))


def replace_dir(new, dest):
    old = dest + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(dest):
        os.replace(dest, old)
    shutil.move(new, dest)
    return old


def copy_source(src, dest):
    say(f"copying the source from {src}")
    patterns = shutil.ignore_patterns(".git", ".dev", "dist", "build", "build-*", "work", "__pycache__", "*.log")

    def ignore(folder, names):  # (and anything that is not a file or a folder: pipes, sockets, devices)
        skipped = set(patterns(folder, names))
        for name in names:
            path = os.path.join(folder, name)
            if not (os.path.isfile(path) or os.path.isdir(path)):
                skipped.add(name)
        return skipped

    tmp = dest + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=ignore)
    shutil.rmtree(replace_dir(tmp, dest), ignore_errors=True)


# ---- game file -----------------------------------------------------------------------------------
def pick_game_file():
    """Asks for the .nup: a file dialog where one is available, else the terminal."""
    prompt = "Select your dump of The Legend of Zelda: Ocarina of Time (Switch 2, .nup; update v131072 is the tested one)"
    path = None
    if not WINDOWS and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        for cmd in (["zenity", "--file-selection", "--title", prompt, "--file-filter=*.nup"],
                    ["kdialog", "--getopenfilename", os.path.expanduser("~"), "*.nup", "--title", prompt]):
            if shutil.which(cmd[0]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                path = r.stdout.strip() or None
                break
    if not path and not WINDOWS:
        try:
            import tkinter, tkinter.filedialog
            root = tkinter.Tk(); root.withdraw()
            path = tkinter.filedialog.askopenfilename(title=prompt, filetypes=[("Switch 2 package", "*.nup"), ("All files", "*")]) or None
            root.destroy()
        except Exception:
            pass
    if not path:
        print(prompt + ".")
        path = input("Path to the .nup file: ").strip().strip('"').strip("'")
    path = os.path.abspath(os.path.expanduser(path)) if path else ""
    if not os.path.isfile(path):
        say("no such file: " + path)
        sys.exit(1)
    return path


def pick_install_dir():
    """Asks where to install (a folder dialog where one is available, else the terminal); nothing chosen:
    the default place."""
    default = default_dir()
    prompt = "Choose where to install TOoT:R (a TOoT-R folder is made there; Cancel: " + default + ")"
    path = None
    if not WINDOWS and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        for cmd in (["zenity", "--file-selection", "--directory", "--title", prompt],
                    ["kdialog", "--getexistingdirectory", os.path.expanduser("~"), "--title", prompt]):
            if shutil.which(cmd[0]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                path = r.stdout.strip() or None
                return os.path.join(path, "TOoT-R") if path else default
        try:
            import tkinter, tkinter.filedialog
            root = tkinter.Tk(); root.withdraw()
            path = tkinter.filedialog.askdirectory(title=prompt) or None
            root.destroy()
            return os.path.join(path, "TOoT-R") if path else default
        except Exception:
            pass
    if sys.stdin and sys.stdin.isatty():
        answer = input(f"Install folder [{default}]: ").strip().strip('"').strip("'")
        return os.path.abspath(os.path.expanduser(answer)) if answer else default
    return default


# ---- the build environment the installer brings (Linux) -----------------------------------------------
# <install folder>/toolchain is a small Linux file system with the compilers, CMake, Ninja, Python and the libraries the
# game links (tools/package/toolchain.sh makes it; the installer's AppImage carries it). The build runs inside it, in a
# sandbox of its own (bubblewrap, which it brings too), so nothing is installed on the PC and no password is asked for:
# what a Steam Deck needs, whose system is read-only. The libraries the game needs at run time are then copied beside
# the executable (app/lib). Without that folder the system's own tools are used, as before.
TOOLCHAIN_GLIBC = (2, 41)
LIBC_FAMILY = ("libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so", "libresolv.so", "libutil.so")
SKIP_LIBS = ("linux-vdso", "ld-linux", "libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so", "libresolv.so",
             "libutil.so", "libvulkan.so", "libGL", "libEGL", "libgbm", "libdrm", "libstdc++.so", "libgcc_s.so")


def toolchain():
    path = os.path.join(ROOT, "toolchain")
    return path if not WINDOWS and os.path.exists(os.path.join(path, "usr", "bin", "clang++")) else None


def sandbox(tool, command, extra_binds=()):
    """command as it runs inside the build environment at `tool`: the install folder is the same folder there."""
    lib = os.path.join(tool, "usr", "lib")
    cmd = [os.path.join(lib, "ld-linux-x86-64.so.2"), "--library-path", lib, os.path.join(tool, "usr", "bin", "bwrap"),
           "--bind", tool, "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--bind", ROOT, ROOT]
    for path in extra_binds:
        cmd += ["--ro-bind", path, path]
    cmd += ["--setenv", "PATH", "/usr/bin", "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8", "--unsetenv", "LD_LIBRARY_PATH",
            "--unsetenv", "PYTHONHOME", "--unsetenv", "PYTHONPATH", "--chdir", ROOT, "--die-with-parent"]
    return cmd + list(command)


def sandbox_works(tool):
    try:
        return subprocess.run(sandbox(tool, ["/usr/bin/true"]), capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def host_glibc():
    """The system's C library version. Asked of the system's own programs: this Python may run on the installer's."""
    import re
    for cmd in (["getconf", "GNU_LIBC_VERSION"], ["ldd", "--version"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
            m = re.search(r"(\d+)\.(\d+)\s*$", out)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return (0, 0)  # (unknown: the game then brings its own, which runs everywhere)


def unpack_toolchain(archive):
    """The installer's build environment into <install folder>/toolchain, with its progress said as it goes."""
    dest = os.path.join(ROOT, "toolchain")
    stamp = os.path.join(dest, ".archive")
    mark = f"{os.path.getsize(archive)}"
    try:
        if open(stamp).read() == mark and toolchain():
            say("the build tools are already unpacked")
            return dest
    except OSError:
        pass
    step("unpack")
    if os.path.isdir(dest):  # (the sandbox makes folders of its own in there; an older one goes whole)
        subprocess.run(["chmod", "-R", "u+rwX", dest], check=False)
        shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    total = os.path.getsize(archive)
    with open(archive, "rb") as raw, tarfile.open(fileobj=raw, mode="r|xz") as t:
        shown = -1
        for member in t:
            if member.name.startswith("/") or ".." in member.name.split("/") or member.isdev():
                continue
            try:
                t.extract(member, dest, filter="tar")
            except TypeError:
                t.extract(member, dest)
            percent = int(raw.tell() * 100 / max(1, total))
            if percent != shown:
                shown = percent
                progress(percent, member.name)
    with open(stamp, "w") as f:
        f.write(mark)
    return dest


def step(key):
    """A line the graphical installers read: which part of the installation begins."""
    print(f"@step {key}", flush=True)


def progress(percent, what=""):
    print(f"@progress {percent} {what}", flush=True)


def bundle_libraries(tool):
    """The libraries the executable loads that came from the build environment, beside it (app/lib), found before the system's."""
    exe = os.path.join(APP, "tootr")
    lib_dir = os.path.join(APP, "lib")
    shutil.rmtree(lib_dir, ignore_errors=True)
    os.makedirs(lib_dir, exist_ok=True)
    listed = subprocess.run(sandbox(tool, ["ldd", exe]), capture_output=True, text=True).stdout
    # A system whose C library is older than the build environment's cannot load the executable by itself: it then
    # starts through the environment's own loader and C library, copied beside it too (cmd_run).
    own_libc = host_glibc() < TOOLCHAIN_GLIBC or os.environ.get("TOOTR_OWN_LIBC") == "1"
    copied = 0
    late = []
    for line in listed.splitlines():
        parts = line.split()
        if "=>" not in parts or len(parts) < 3 or not parts[2].startswith("/"):
            continue
        name, inside = os.path.basename(parts[0]), parts[2]
        if inside.startswith(lib_dir) or name.startswith(("ld-linux", "linux-vdso")):
            continue
        source = os.path.realpath(os.path.join(tool, inside.lstrip("/")))
        if not source.startswith(tool) or not os.path.isfile(source):
            continue
        if name.startswith(("libstdc++.so", "libgcc_s.so")):
            late.append((name, source))
        if own_libc and name.startswith(LIBC_FAMILY):
            shutil.copyfile(source, os.path.join(lib_dir, name)); copied += 1
            continue
        if name.startswith(SKIP_LIBS):
            continue
        shutil.copyfile(source, os.path.join(lib_dir, name)); copied += 1
    subprocess.run(sandbox(tool, ["patchelf", "--force-rpath", "--set-rpath", "$ORIGIN/lib", exe]), check=False)
    # The C++ runtime is the system's when it is new enough (its graphics drivers need at least their own), else the
    # build environment's.
    cfg = load_config(); cfg["own_libc"] = own_libc; save_config(cfg)
    if own_libc:
        loader = os.path.join(lib_dir, "ld-linux-x86-64.so.2")
        shutil.copyfile(os.path.realpath(os.path.join(tool, "usr", "lib", "ld-linux-x86-64.so.2")), loader)
        os.chmod(loader, 0o755)
        for name, source in late:
            shutil.copyfile(source, os.path.join(lib_dir, name)); copied += 1
        ok = subprocess.run([loader, "--library-path", lib_dir, "--list", exe], capture_output=True, text=True)
        missing = [l.strip() for l in (ok.stdout + ok.stderr).splitlines() if "not found" in l]
        say(f"{copied} libraries placed beside the game, with their own C library (this system's is older than the build tools')"
            + (f"; still missing: {'; '.join(missing[:6])}" if missing else ""))
        return not missing
    check = subprocess.run(["ldd", exe], capture_output=True, text=True)
    if "not found" in check.stdout + check.stderr:
        for name, source in late:
            shutil.copyfile(source, os.path.join(lib_dir, name)); copied += 1
        check = subprocess.run(["ldd", exe], capture_output=True, text=True)
    missing = [l.strip() for l in (check.stdout + check.stderr).splitlines() if "not found" in l]
    say(f"{copied} libraries placed beside the game" + (f"; still missing on this system: {'; '.join(missing[:6])}" if missing else ""))
    return not missing


# ---- build -------------------------------------------------------------------------------------
def fetch_dlss(source):
    """NVIDIA's DLSS SDK, downloaded from NVIDIA's own repository onto this PC (tools/ngx_sdk.py: a pinned release, every
    file checked against its SHA-256). It is never part of this project or of anything this installer ships: NVIDIA's files
    are under NVIDIA's licence, which the user accepted in the installer (the "--dlss" option says so). The headers and
    the library go to <install>/build/ngx-sdk for the build; DLSS's runtime libraries to <install>/ngx, where the game
    looks for them. Returns the SDK folder, or None when the download failed (the build then leaves DLSS out)."""
    sdk = os.path.join(ROOT, "build", "ngx-sdk")
    step("dlss")
    say("NVIDIA DLSS: downloading NVIDIA's DLSS SDK from github.com/NVIDIA/DLSS (NVIDIA's licence applies to it)")
    cmd = [sys.executable, "-I", os.path.join(source, "tools", "ngx_sdk.py"), "--accept-nvidia-license",
           "--platform", "windows" if WINDOWS else "linux", "--runtime-dir", os.path.join(ROOT, "ngx"), sdk]
    try:
        ok = subprocess.run(cmd).returncode == 0
    except OSError:
        ok = False
    if not ok:
        say("NVIDIA DLSS: the download did not finish: the game is built without DLSS (AMD FSR 3.1 still works). Install again to retry.")
        return None
    return sdk


def build(source, nup, dlss=False):
    args = [os.path.join(source, "tools", "build_game.py"), "--data", ROOT, "--prefix", APP, "--work", os.path.join(ROOT, "build")]
    sdk = fetch_dlss(source) if dlss else None
    os.environ["OOT_NGX_SDK"] = sdk or ""  # (tools/build_game.py passes it to CMake)
    if nup:
        args += ["--nup", nup]
    tool = toolchain()
    if tool and not sandbox_works(tool):
        say("the build tools' sandbox cannot start on this system (unprivileged user namespaces are off): using the system's tools")
        tool = None
    binds = [nup] if tool and nup and os.path.exists(nup) and not os.path.abspath(nup).startswith(ROOT + os.sep) else []
    def attempt():
        return run_logged(sandbox(tool, ["python3"] + args, binds) if tool else [sys.executable] + args, "build") == 0
    ok = attempt()
    if not ok and sdk:
        # (why, in the launcher's own log: its build log is replaced by the next attempt's - kept as build.1.log)
        try:
            with open(os.path.join(log_dir() or "", "build.log"), errors="replace") as f:
                lines = [l.rstrip() for l in f if "error" in l.lower() or "undefined" in l.lower()]
            for line in lines[:12]:
                say("  " + line[:300])
        except OSError:
            pass
        # (the game itself must not be lost to an optional part: once more without NVIDIA's library)
        say("the build with NVIDIA DLSS did not finish: building once more without it (AMD FSR 3.1 still works)")
        os.environ["OOT_NGX_SDK"] = ""
        ok = attempt()
    if not ok:
        say("the build's output is kept in " + os.path.join(ROOT, "logs", "build.log"))
        return False
    if tool:
        step("libraries")
        return bundle_libraries(tool)
    return True


def write_shortcuts(source):
    if WINDOWS:
        return  # install.ps1 creates the Start menu and desktop shortcuts
    bin_dir = os.path.join(os.path.expanduser("~"), ".local", "bin")
    os.makedirs(bin_dir, exist_ok=True)
    launcher = os.path.join(bin_dir, "tootr")
    with open(launcher, "w") as f:
        tool = os.path.join(ROOT, "toolchain", "usr")
        f.write("#!/bin/sh\n# tootr: checks for updates, then starts the game.\n"
                f'export OOT_RECOMP_HOME="{ROOT}"\n'
                f'launcher="{os.path.join(source, "installer", "launcher.py")}"\n'
                'if command -v python3 >/dev/null 2>&1; then exec python3 "$launcher" run "$@"; fi\n'
                "# (no Python on this system: the one of the build tools the installer brought)\n"
                f'PYTHONHOME="{tool}" exec "{tool}/lib/ld-linux-x86-64.so.2" --library-path "{tool}/lib" "{tool}/bin/python3" "$launcher" run "$@"\n')
    os.chmod(launcher, 0o755)
    apps = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share"), "applications")
    os.makedirs(apps, exist_ok=True)
    with open(os.path.join(apps, "tootr.desktop"), "w") as f:
        icon = os.path.join(source, "installer", "tootr.png")
        f.write("[Desktop Entry]\nType=Application\nName=TOoT:R\n"
                "Comment=Native PC port built from your own copy of the game\n"
                f"Exec={launcher}\nIcon={icon}\nTerminal=false\nCategories=Game;\n")
    say(f"installed the launcher: {launcher} (and an application menu entry)")


def cmd_install(args):
    nup = None
    src = None
    target = None
    tool_archive = None
    dlss = None  # (None: as the installation was made before)
    i = 0
    while i < len(args):
        if args[i] == "--nup":
            nup = args[i + 1]; i += 2
        elif args[i] == "--source":
            src = args[i + 1]; i += 2
        elif args[i] == "--dir":
            target = args[i + 1]; i += 2
        elif args[i] == "--toolchain":
            tool_archive = args[i + 1]; i += 2
        elif args[i] == "--dlss":  # (NVIDIA DLSS: its SDK is downloaded from NVIDIA; the user accepted NVIDIA's licence)
            dlss = True; i += 1
        elif args[i] == "--no-dlss":
            dlss = False; i += 1
        else:
            say("unknown option " + args[i]); sys.exit(2)
    if not nup:
        cfg_here = load_config()
        if not (cfg_here.get("nup") and os.path.isfile(cfg_here["nup"])):
            nup = pick_game_file()
    if target is None and not os.environ.get("OOT_RECOMP_HOME"):
        target = pick_install_dir()
    if target:
        set_root(target)
    say("installing into " + ROOT)
    os.makedirs(ROOT, exist_ok=True)
    cfg = load_config()
    nup = os.path.abspath(os.path.expanduser(nup)) if nup else (cfg.get("nup") if cfg.get("nup") and os.path.isfile(cfg["nup"]) else pick_game_file())
    if not os.path.isfile(nup):
        say("no such file: " + nup); sys.exit(1)
    cfg["nup"] = nup
    save_config(cfg)
    src = os.path.abspath(src or os.path.dirname(HERE))
    if os.path.abspath(src) != os.path.abspath(SOURCE):
        copy_source(src, SOURCE)
    slug = repo_slug(SOURCE)
    if slug and not cfg.get("commit"):
        try:
            cfg["commit"] = commit_of_tree(src) or latest_commit(slug)
        except Exception:
            pass
    if tool_archive and os.path.isfile(tool_archive):
        unpack_toolchain(tool_archive)
    say("building (the first build compiles the whole game: this can take 15-60 minutes)")
    if dlss is not None:
        cfg["dlss"] = bool(dlss); save_config(cfg)
    if not build(SOURCE, nup, bool(cfg.get("dlss"))):
        say("the build failed; see the messages above")
        sys.exit(1)
    save_config(cfg)
    step("shortcuts")
    write_shortcuts(SOURCE)
    step("done")
    say("installed into " + ROOT + ". Start the game with: " + ("the Start menu or desktop shortcut" if WINDOWS else "tootr"))


def commit_of_tree(path):
    if shutil.which("git") and os.path.isdir(os.path.join(path, ".git")):
        r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    return None


def update(force=False):
    """Returns True when an update was installed."""
    if os.environ.get("OOT_RECOMP_NO_UPDATE") == "1":
        return False
    cfg = load_config()
    slug = repo_slug(SOURCE)
    if not slug:
        return False
    try:
        sha = latest_commit(slug)
    except Exception as e:
        if force:
            say(f"cannot reach GitHub ({e})")
        return False
    if sha == cfg.get("commit") and not force:
        return False
    say(f"update available ({(cfg.get('commit') or 'unknown')[:10]} -> {sha[:10]}); installing")
    notify("Installing an update; the game starts when it is built (usually a few minutes).")
    try:
        download_source(slug, sha, SOURCE)
    except Exception as e:
        say(f"download failed ({e}); starting the installed version")
        return False
    if build(SOURCE, cfg.get("nup"), bool(cfg.get("dlss"))):  # (DLSS as the installation chose it)
        shutil.rmtree(SOURCE + ".old", ignore_errors=True)
        cfg["commit"] = sha
        cfg["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_config(cfg)
        say("update installed")
        return True
    say("the update did not build; going back to the installed version")
    notify("The update did not build; starting the installed version.")
    if os.path.exists(SOURCE + ".old"):
        shutil.rmtree(SOURCE, ignore_errors=True)
        os.replace(SOURCE + ".old", SOURCE)
        # (objects of the newer sources must not survive into the next build)
        shutil.rmtree(os.path.join(ROOT, "build", "runtime-build"), ignore_errors=True)
    return False


def cmd_run(args):
    # (tootr --game [file]: choose the copy of the game to play from, then play)
    if args and args[0] in ("--game", "--choose-game"):
        path = args[1:2] if len(args) > 1 and not args[1].startswith("-") else []
        if cmd_set_game(path):
            return 1
        args = args[1 + len(path):]
    update()
    cfg = load_config()
    exe = os.path.join(APP, "tootr" + EXE)
    if not os.path.exists(exe):
        say("not installed yet: run the installer first")
        sys.exit(1)
    env = dict(os.environ)
    if cfg.get("nup"):
        env.setdefault("OOT_RECOMP_NUP", cfg["nup"])
    env.setdefault("OOT_RECOMP_DATA", ROOT)
    command = [exe] + args
    loader = os.path.join(APP, "lib", "ld-linux-x86-64.so.2")
    if not WINDOWS and cfg.get("own_libc") and os.path.exists(loader):  # (built against a newer C library than this system's: bundle_libraries)
        command = [loader, "--library-path", os.path.join(APP, "lib"), exe] + args
    say("starting the game: its log is " + os.path.join(ROOT, "logs", "tootr.log"))
    try:
        code = subprocess.run(command, env=env).returncode
    except KeyboardInterrupt:
        return 130
    except OSError as e:
        notify("TOoT:R could not be started: %s\n\nThe logs are in %s" % (e, os.path.join(ROOT, "logs")))
        return 1
    # (ended by itself with an error, or by a fault: the player is told where the log is. Closed by the player or the
    # system - a signal such as SIGTERM, Ctrl+C - is not an error. The game shows its own message for the errors it
    # knows; this is for the ones it could not: a missing library, a crash in a driver.)
    stopped = code in (-2, -15, -9, 130, 143) or (WINDOWS and code in (0xC000013A, -1073741510))
    if code != 0 and not stopped:
        log = os.path.join(ROOT, "logs", "tootr.log")
        tail = ""
        try:
            with open(log, "rb") as f:
                lines = f.read().decode("utf-8", "replace").splitlines()
            fatal = [l for l in lines if "FATAL" in l or "access violation" in l or "SIGSEGV" in l]
            tail = "\n\n" + fatal[-1][:400] if fatal else ""
        except OSError:
            pass
        shown_by_game = bool(tail)  # (the game's own message box said it: only noted here)
        text = "TOoT:R closed unexpectedly (code %s).%s\n\nThe log of this run is kept here; attach it when you report the problem:\n%s" % (
            hex(code & 0xffffffff) if WINDOWS else code, tail, log)
        if shown_by_game:
            say(text.replace("\n", " "))
        else:
            notify(text)
    return code


def cmd_set_game(args):
    """Another copy of the game for this installation: asked for when no path is given. Its code is read at once; a
    copy of the same game version changes nothing more, another version's is built (the long step)."""
    cfg = load_config()
    before = cfg.get("nup")
    nup = os.path.abspath(os.path.expanduser(args[0])) if args else pick_game_file()
    if not os.path.isfile(nup):
        say("no such file: " + nup)
        return 1
    cfg["nup"] = nup
    save_config(cfg)
    say("game file: " + nup)
    if os.path.isdir(SOURCE) and not build(SOURCE, nup, bool(cfg.get("dlss"))):
        cfg["nup"] = before
        save_config(cfg)
        say("this file could not be built: the game file is as it was" + (" (" + before + ")" if before else ""))
        return 1
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "install":
        cmd_install(args)
    elif cmd == "run":
        return cmd_run(args)
    elif cmd == "update":
        update(force="--force" in args) or say("already up to date")
    elif cmd == "set-game":
        cmd_set_game(args)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
