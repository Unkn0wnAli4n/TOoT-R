#!/usr/bin/env python3
"""TOoT:R launcher, installer back end and auto-updater (Linux and Windows/MSYS2).

    launcher.py install [--nup PATH] [--dir DIR] [--source DIR]
                                                       first install (asks for the game file and where
                                                       to install; the default place when none is given)
    launcher.py run [game args...]                     update if GitHub has a newer version, then play
    launcher.py update [--force]                       only update
    launcher.py set-game [PATH]                        point the install at another copy of the game

Layout under the install directory (by default ~/.local/share/tootr, or %LOCALAPPDATA%\\tootr; the
launcher finds the one it was installed into, OOT_RECOMP_HOME overrides):
    source/     this project's source (replaced on update)
    app/        the built executable
    build/      build cache (offline tools, the compiled game code archive, runtime build)
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


def say(msg):
    print(f"[TOoT:R] {msg}", flush=True)


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
    ignore = shutil.ignore_patterns(".git", ".dev", "build", "build-*", "work", "__pycache__", "*.log")
    tmp = dest + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=ignore)
    shutil.rmtree(replace_dir(tmp, dest), ignore_errors=True)


# ---- game file -----------------------------------------------------------------------------------
def pick_game_file():
    """Asks for the .nup: a file dialog where one is available, else the terminal."""
    prompt = "Select your dump of The Legend of Zelda: Ocarina of Time (Switch 2, v131072, .nup)"
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


# ---- build -------------------------------------------------------------------------------------
def build(source, nup):
    cmd = [sys.executable, os.path.join(source, "tools", "build_game.py"),
           "--data", ROOT, "--prefix", APP, "--work", os.path.join(ROOT, "build")]
    if nup:
        cmd += ["--nup", nup]
    return subprocess.run(cmd).returncode == 0


def write_shortcuts(source):
    if WINDOWS:
        return  # install.ps1 creates the Start menu and desktop shortcuts
    bin_dir = os.path.join(os.path.expanduser("~"), ".local", "bin")
    os.makedirs(bin_dir, exist_ok=True)
    launcher = os.path.join(bin_dir, "tootr")
    with open(launcher, "w") as f:
        f.write("#!/bin/sh\n# tootr: checks for updates, then starts the game.\n"
                f'OOT_RECOMP_HOME="{ROOT}" exec python3 "{os.path.join(source, "installer", "launcher.py")}" run "$@"\n')
    os.chmod(launcher, 0o755)
    apps = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share"), "applications")
    os.makedirs(apps, exist_ok=True)
    with open(os.path.join(apps, "tootr.desktop"), "w") as f:
        f.write("[Desktop Entry]\nType=Application\nName=TOoT:R\n"
                "Comment=Native PC port built from your own copy of the game\n"
                f"Exec={launcher}\nTerminal=false\nCategories=Game;\n")
    say(f"installed the launcher: {launcher} (and an application menu entry)")


def cmd_install(args):
    nup = None
    src = None
    target = None
    i = 0
    while i < len(args):
        if args[i] == "--nup":
            nup = args[i + 1]; i += 2
        elif args[i] == "--source":
            src = args[i + 1]; i += 2
        elif args[i] == "--dir":
            target = args[i + 1]; i += 2
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
    say("building (the first build compiles the whole game: this can take 15-60 minutes)")
    if not build(SOURCE, nup):
        say("the build failed; see the messages above")
        sys.exit(1)
    save_config(cfg)
    write_shortcuts(SOURCE)
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
    if build(SOURCE, cfg.get("nup")):
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
    if WINDOWS:
        return subprocess.run([exe] + args, env=env).returncode
    os.execve(exe, [exe] + args, env)


def cmd_set_game(args):
    cfg = load_config()
    cfg["nup"] = os.path.abspath(args[0]) if args else pick_game_file()
    save_config(cfg)
    say("game file: " + cfg["nup"])


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
