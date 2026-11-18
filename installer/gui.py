#!/usr/bin/env python3
"""TOoT:R graphical installer (Linux; packaged as an AppImage by tools/package/appimage.sh).

Three pages: choose your own copy of the game and where to install; watch the installation, step by step, with what is
being worked on and how long is left; play. The build needs compilers and libraries: the AppImage brings them
(TOOTR_TOOLCHAIN, a file system unpacked into the install folder and used in a sandbox), so nothing is installed on the
system and no password is asked for - which is what a Steam Deck needs. Where that cannot work (a system that
allows no unprivileged sandboxes) the system's packages are installed instead, through the administrator prompt.
The work itself is installer/launcher.py's; this window reads its output (the "@step" and "@progress" lines, and the
compiler's "[n/m]" ones).
"""
import os, queue, re, shutil, subprocess, sys, threading, time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.environ.get("TOOTR_SOURCE") or os.path.dirname(HERE)
TOOLCHAIN = os.environ.get("TOOTR_TOOLCHAIN") or ""
DEFAULT_DIR = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "tootr")
NEEDED_GB = 10  # (the build's objects, the build tools and the game's caches)

DEPS = {
    "pacman": ["pacman", "-S", "--needed", "--noconfirm", "clang", "lld", "cmake", "ninja", "python", "pkgconf", "sdl3",
               "opus", "ffmpeg", "vulkan-icd-loader", "vulkan-headers", "fmt", "zstd", "lz4", "jemalloc", "curl"],
    "apt-get": ["sh", "-c", "apt-get update && apt-get install -y clang lld cmake ninja-build python3 pkg-config libsdl3-dev "
                "libopus-dev libavcodec-dev libavutil-dev libswscale-dev libvulkan-dev libfmt-dev libzstd-dev liblz4-dev "
                "libjemalloc-dev curl"],
    "dnf": ["dnf", "install", "-y", "clang", "lld", "cmake", "ninja-build", "python3", "pkgconf-pkg-config", "SDL3-devel",
            "opus-devel", "ffmpeg-free-devel", "vulkan-loader-devel", "vulkan-headers", "fmt-devel", "libzstd-devel",
            "lz4-devel", "jemalloc-devel", "curl"],
    "zypper": ["zypper", "install", "-y", "clang", "lld", "cmake", "ninja", "python3", "pkg-config", "SDL3-devel",
               "libopus-devel", "ffmpeg-7-libavcodec-devel", "ffmpeg-7-libavutil-devel", "ffmpeg-7-libswscale-devel",
               "vulkan-devel", "fmt-devel", "libzstd-devel", "liblz4-devel", "jemalloc-devel", "curl"],
}

# The installation's parts: key (what launcher.py and build_game.py name in "@step"), title, share of the whole.
STEPS = [
    ("packages", "Install the build tools", 4),
    ("unpack", "Unpack the build tools", 5),
    ("tools", "Prepare the translator", 3),
    ("dlss", "Download NVIDIA DLSS (from NVIDIA)", 1),
    ("check", "Read and check your game", 3),
    ("translate", "Translate the game's code", 6),
    ("compile", "Compile the game", 62),
    ("runtime", "Build TOoT:R", 13),
    ("libraries", "Add the libraries it needs", 2),
    ("shortcuts", "Create the shortcuts", 2),
]

BG, PANEL, FIELD, TEXT, MUTED, ACCENT, GOOD, BAD = "#12151c", "#1b202b", "#242b39", "#e8eaf0", "#8d96a8", "#e0b64a", "#6fcf7f", "#ef6b6b"


def glibc():
    try:
        return tuple(int(x) for x in os.confstr("CS_GNU_LIBC_VERSION").split()[1].split(".")[:2])
    except (ValueError, OSError, AttributeError, IndexError):
        return (0, 0)


def sandboxes_allowed():
    """Whether a program may make a sandbox of its own here (unprivileged user namespaces), as the bundled tools need."""
    for path, bad in (("/proc/sys/kernel/unprivileged_userns_clone", "0"), ("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", "1"),
                      ("/proc/sys/user/max_user_namespaces", "0")):
        try:
            if open(path).read().strip() == bad:
                return False
        except OSError:
            pass
    return True


def bundled_tools():
    """None when the installer's own build tools can be used, else why not."""
    if not TOOLCHAIN or not os.path.isfile(TOOLCHAIN):
        return "this installer was made without them"
    if not sandboxes_allowed():
        return "this system does not let programs make sandboxes"
    return None


def find_games():
    """.nup files in the places a dump usually ends up, newest first."""
    found = []
    home = os.path.expanduser("~")
    places = [home] + [os.path.join(home, d) for d in ("Downloads", "Desktop", "Documents", "Games", "ROMs", "roms")]
    places += [os.path.join("/run/media", d) for d in (os.listdir("/run/media") if os.path.isdir("/run/media") else [])]
    for place in places:
        try:
            for depth_root, dirs, files in os.walk(place):
                if depth_root.count(os.sep) - place.count(os.sep) >= 2:
                    dirs[:] = []
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                found += [os.path.join(depth_root, f) for f in files if f.lower().endswith(".nup")]
        except OSError:
            pass
    unique = sorted(set(found), key=lambda p: -os.path.getmtime(p))
    return unique[:8]


def size_text(n):
    return "%.1f GB" % (n / 2**30) if n >= 2**30 else "%.0f MB" % (n / 2**20)


def duration(seconds):
    seconds = int(max(0, seconds))
    return "%d min" % round(seconds / 60) if seconds >= 90 else "%d s" % seconds


class Installer(tk.Tk):
    def __init__(self):
        super().__init__(className="tootr-installer")
        self.title("TOoT:R Installer")
        self.configure(bg=BG)
        w, h = min(900, self.winfo_screenwidth() - 40), min(680, self.winfo_screenheight() - 80)
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth() - w) // 2}+{max(0, (self.winfo_screenheight() - h) // 2 - 20)}")
        self.minsize(720, 520)
        try:
            self.icon = tk.PhotoImage(file=os.path.join(HERE, "tootr.png"))
            self.iconphoto(True, self.icon)
        except tk.TclError:
            self.icon = None
        self.lines = queue.Queue()
        self.process = None
        self.target = DEFAULT_DIR
        self.why_not_bundled = bundled_tools()
        self.state_of = {}
        self.started = 0.0
        self.step_started = 0.0
        self.current = None
        self.fraction = 0.0
        self.style()

        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x")
        if self.icon:
            self.small_icon = self.icon.subsample(max(1, round(self.icon.height() / 64)))
            tk.Label(header, image=self.small_icon, bg=PANEL).pack(side="left", padx=(22, 14), pady=14)
        titles = tk.Frame(header, bg=PANEL)
        titles.pack(side="left", pady=14, padx=(0 if self.icon else 22, 0))
        tk.Label(titles, text="TOoT:R", bg=PANEL, fg=ACCENT, font=("TkDefaultFont", 22, "bold")).pack(anchor="w")
        tk.Label(titles, text="A native PC version of the game, built on this computer from your own copy.", bg=PANEL, fg=MUTED).pack(anchor="w")
        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True, padx=26, pady=18)
        self.setup_page()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(100, self.pump)

    def style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, fieldbackground=FIELD, bordercolor=FIELD, lightcolor=FIELD, darkcolor=FIELD, troughcolor=FIELD, focuscolor=ACCENT)
        s.configure("TButton", background=FIELD, foreground=TEXT, padding=(16, 9), borderwidth=0)
        s.map("TButton", background=[("disabled", PANEL), ("active", "#323b4d")], foreground=[("disabled", MUTED)])
        s.configure("Go.TButton", background=ACCENT, foreground="#1a1407", font=("TkDefaultFont", 12, "bold"), padding=(28, 12))
        s.map("Go.TButton", background=[("disabled", "#5a4d2a"), ("active", "#f0c966")], foreground=[("disabled", "#2a2416")])
        s.configure("TEntry", foreground=TEXT, insertcolor=TEXT, padding=7)
        s.configure("TCheckbutton", background=BG, foreground=TEXT)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=FIELD, bordercolor=FIELD, lightcolor=ACCENT, darkcolor=ACCENT, thickness=16)

    def clear(self):
        for child in self.body.winfo_children():
            child.destroy()

    def label(self, parent, text, color=TEXT, size=None, bold=False, **kw):
        font = ("TkDefaultFont", size or 10, "bold" if bold else "normal")
        return tk.Label(parent, text=text, bg=parent["bg"], fg=color, font=font, anchor="w", justify="left", **kw)

    # ---- page 1: what and where ---------------------------------------------------------------------
    def setup_page(self):
        self.clear()
        b = self.body
        self.label(b, "1   Your copy of the game", ACCENT, 12, True).pack(anchor="w")
        self.label(b, "The Legend of Zelda: Ocarina of Time for Nintendo Switch 2, dumped as one .nup file. Update v131072 is the tested one; "
                      "other updates are built from what is found in them, and may lack some options. "
                      "It is only read on this computer: nothing from it is uploaded or downloaded.", MUTED, wraplength=820).pack(anchor="w", pady=(2, 8))
        row = tk.Frame(b, bg=BG); row.pack(fill="x")
        self.nup = tk.StringVar()
        ttk.Entry(row, textvariable=self.nup).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self.pick_nup).pack(side="left", padx=(8, 0))
        self.nup_note = self.label(b, "", MUTED); self.nup_note.pack(anchor="w", pady=(4, 0))
        games = find_games()
        if games:
            self.nup.set(games[0])
            if len(games) > 1:
                found = tk.Frame(b, bg=BG); found.pack(fill="x", pady=(4, 0))
                self.label(found, "Also found:", MUTED).pack(side="left")
                for path in games[1:4]:
                    link = self.label(found, os.path.basename(path), ACCENT, cursor="hand2"); link.pack(side="left", padx=(8, 0))
                    link.bind("<Button-1>", lambda _e, p=path: self.nup.set(p))
        self.nup.trace_add("write", lambda *_: self.check())

        self.label(b, "2   Where to install", ACCENT, 12, True).pack(anchor="w", pady=(20, 0))
        self.label(b, "The game, its saves and its settings all live in this folder.", MUTED).pack(anchor="w", pady=(2, 8))
        row = tk.Frame(b, bg=BG); row.pack(fill="x")
        self.folder = tk.StringVar(value=DEFAULT_DIR)
        ttk.Entry(row, textvariable=self.folder).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Change…", command=self.pick_folder).pack(side="left", padx=(8, 0))
        self.space_note = self.label(b, "", MUTED); self.space_note.pack(anchor="w", pady=(4, 0))
        self.folder.trace_add("write", lambda *_: self.check())

        self.label(b, "3   Build tools", ACCENT, 12, True).pack(anchor="w", pady=(20, 0))
        self.deps = tk.BooleanVar(value=True)
        if not self.why_not_bundled:
            self.label(b, "✓  Included with this installer. Nothing is installed on your system and no password is needed "
                          "(this also works on a Steam Deck).", GOOD, wraplength=820).pack(anchor="w", pady=(2, 0))
        else:
            self.label(b, f"The installer's own build tools cannot be used here: {self.why_not_bundled}. "
                          "Your system's packages are used instead.", MUTED, wraplength=820).pack(anchor="w", pady=(2, 4))
            ttk.Checkbutton(b, text="Install them now (asks for your password)", variable=self.deps).pack(anchor="w")

        # NVIDIA DLSS is never part of this installer: when asked for, NVIDIA's SDK is downloaded from NVIDIA's own repository
        # onto this PC during the installation (installer/launcher.py fetch_dlss), under NVIDIA's licence.
        self.label(b, "4   Optional: NVIDIA DLSS", ACCENT, 12, True).pack(anchor="w", pady=(20, 0))
        self.dlss = tk.BooleanVar(value=False)
        ttk.Checkbutton(b, text="Add NVIDIA DLSS, DLAA and DLSS Frame Generation (NVIDIA RTX graphics cards)", variable=self.dlss).pack(anchor="w", pady=(2, 0))
        self.label(b, "Downloads NVIDIA's DLSS SDK (about 60 MB) from github.com/NVIDIA/DLSS during the installation. It is NVIDIA's software, "
                      "not part of TOoT:R: by ticking the box you accept NVIDIA's RTX SDKs licence (github.com/NVIDIA/DLSS/blob/main/LICENSE.txt). "
                      "Without it, AMD FSR 3.1 upscaling and frame generation still work on every graphics card.", MUTED, wraplength=820).pack(anchor="w")

        bottom = tk.Frame(b, bg=BG); bottom.pack(side="bottom", fill="x")
        self.go = ttk.Button(bottom, text="Install", style="Go.TButton", command=self.start)
        self.go.pack(side="right")
        ttk.Button(bottom, text="Close", command=self.close).pack(side="right", padx=(0, 10))
        self.label(bottom, "The first installation compiles the whole game:\n15 to 60 minutes, depending on the processor.", MUTED).pack(side="left")
        self.check()

    def pick_nup(self):
        path = filedialog.askopenfilename(title="Your dump of the game", filetypes=[("Switch 2 package", "*.nup"), ("All files", "*")])
        if path:
            self.nup.set(path)

    def pick_folder(self):
        path = filedialog.askdirectory(title="Install into (a TOoT-R folder is made there)")
        if path:
            self.folder.set(os.path.join(path, "TOoT-R"))

    def check(self):
        nup = os.path.expanduser(self.nup.get().strip())
        ok = os.path.isfile(nup)
        if not nup:
            self.nup_note.configure(text="Choose your .nup file.", fg=MUTED)
        elif not ok:
            self.nup_note.configure(text="✗  There is no such file.", fg=BAD)
        else:
            size = os.path.getsize(nup)
            small = size < 2**30
            self.nup_note.configure(text=("✗  %s is too small to be the game (%s)." if small else "✓  %s  (%s). It is checked exactly when the installation starts.")
                                    % (os.path.basename(nup), size_text(size)), fg=BAD if small else GOOD)
            ok = not small
        folder = os.path.abspath(os.path.expanduser(self.folder.get().strip() or DEFAULT_DIR))
        probe = folder
        while probe and not os.path.isdir(probe):
            probe = os.path.dirname(probe)
        try:
            free = shutil.disk_usage(probe or "/").free
            enough = free >= NEEDED_GB * 2**30 or os.path.isdir(os.path.join(folder, "build"))
            writable = os.access(probe or "/", os.W_OK)
            if not writable:
                self.space_note.configure(text="✗  This folder cannot be written to.", fg=BAD)
            else:
                self.space_note.configure(text=("✓  %s free (about %d GB are used while building, 3 GB afterwards)." if enough else
                                                "✗  Only %s free: about %d GB are needed while building.") % (size_text(free), NEEDED_GB), fg=GOOD if enough else BAD)
            ok = ok and enough and writable
        except OSError:
            pass
        self.go.configure(state="normal" if ok else "disabled")

    # ---- page 2: the installation --------------------------------------------------------------------
    def progress_page(self):
        self.clear()
        b = self.body
        self.heading = self.label(b, "Installing", TEXT, 15, True); self.heading.pack(anchor="w")
        self.label(b, "Into " + self.target, MUTED).pack(anchor="w", pady=(0, 12))
        steps = tk.Frame(b, bg=BG); steps.pack(fill="x")
        self.rows = {}
        for key, title, _ in STEPS:
            if key == "packages" and (not self.why_not_bundled or not self.deps.get()):
                continue
            if key in ("unpack", "libraries") and self.why_not_bundled:
                continue
            row = tk.Frame(steps, bg=BG); row.pack(fill="x", pady=2)
            mark = self.label(row, "○", MUTED, 12, width=2); mark.pack(side="left")
            name = self.label(row, title, MUTED, 11); name.pack(side="left")
            note = self.label(row, "", MUTED); note.pack(side="right")
            self.rows[key] = (mark, name, note)
        self.bar = ttk.Progressbar(b, mode="determinate", maximum=1000)
        self.bar.pack(fill="x", pady=(18, 6))
        line = tk.Frame(b, bg=BG); line.pack(fill="x")
        self.percent = self.label(line, "0%", TEXT, 11, True); self.percent.pack(side="left")
        self.remaining = self.label(line, "", MUTED); self.remaining.pack(side="right")
        self.doing = self.label(b, "Starting…", MUTED); self.doing.pack(fill="x", pady=(6, 0))

        bottom = tk.Frame(b, bg=BG); bottom.pack(side="bottom", fill="x")
        self.cancel = ttk.Button(bottom, text="Cancel", command=self.close); self.cancel.pack(side="right")
        self.details_shown = False
        self.toggle = ttk.Button(bottom, text="Show details", command=self.toggle_details); self.toggle.pack(side="left")
        self.log_frame = tk.Frame(b, bg=BG)
        self.log = tk.Text(self.log_frame, height=8, state="disabled", wrap="none", font=("TkFixedFont", 9), bg=PANEL, fg=MUTED,
                           relief="flat", highlightthickness=0, padx=8, pady=6)
        self.log.pack(fill="both", expand=True)

    def toggle_details(self):
        self.details_shown = not self.details_shown
        if self.details_shown:
            self.log_frame.pack(fill="both", expand=True, pady=(12, 12))
            self.log.see("end")
        else:
            self.log_frame.pack_forget()
        self.toggle.configure(text="Hide details" if self.details_shown else "Show details")

    def write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        if int(self.log.index("end-1c").split(".")[0]) > 4000:
            self.log.delete("1.0", "1000.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def begin(self, key):
        """A part begins: the ones before it are done (also those that had nothing to do)."""
        keys = [k for k, _, _ in STEPS]
        if key not in keys:
            if key == "done":
                key = None
            else:
                return
        for k in keys:
            if key is not None and keys.index(k) >= keys.index(key):
                break
            if k in self.rows and self.state_of.get(k) != "done":
                mark, name, note = self.rows[k]
                mark.configure(text="✓", fg=GOOD); name.configure(fg=TEXT, font=("TkDefaultFont", 11, "normal"))
                if self.state_of.get(k) == "running":
                    note.configure(text=duration(time.time() - self.step_started))
                self.state_of[k] = "done"
        self.current = key
        self.fraction = 0.0
        self.step_started = time.time()
        if key in self.rows:
            mark, name, _ = self.rows[key]
            mark.configure(text="●", fg=ACCENT); name.configure(fg=TEXT, font=("TkDefaultFont", 11, "bold"))
            self.state_of[key] = "running"
        self.show_progress()

    def show_progress(self):
        present = [(k, w) for k, _, w in STEPS if k in self.rows]
        total = sum(w for _, w in present) or 1
        done = sum(w for k, w in present if self.state_of.get(k) == "done")
        now = next((w for k, w in present if k == self.current), 0) * self.fraction
        overall = (done + now) / total
        self.bar.configure(value=int(overall * 1000))
        self.percent.configure(text="%d%%" % int(overall * 100))
        elapsed = time.time() - self.started
        if overall > 0.08 and elapsed > 45:
            self.remaining.configure(text="about %s left  ·  %s so far" % (duration(elapsed * (1 - overall) / overall), duration(elapsed)))
        else:
            self.remaining.configure(text="%s so far" % duration(elapsed))

    def read(self, line):
        """One line of the installation's output: the parts and their progress are shown, everything goes to the details."""
        text = line.rstrip("\n")
        if text.startswith("@step "):
            self.begin(text[6:].strip())
            return
        if text.startswith("@progress "):
            parts = text.split(" ", 2)
            try:
                self.fraction = int(parts[1]) / 100
            except ValueError:
                pass
            if len(parts) > 2 and parts[2]:
                self.doing.configure(text=parts[2][-110:])
            self.show_progress()
            return
        m = re.match(r"\[(\d+)/(\d+)\] (.*)", text)
        if m:
            n, of = int(m.group(1)), int(m.group(2))
            self.fraction = n / max(1, of)
            what = m.group(3)
            file = re.search(r"([\w.+-]+)\.(?:cpp|c|cc)\.o", what)
            where = "the game's code" if self.current == "compile" else "TOoT:R" if self.current == "runtime" else "the translator"
            self.doing.configure(text=("Compiling %s: %s  (file %d of %d)" % (where, file.group(1), n, of)) if file else ("%s  (%d of %d)" % (what[:80], n, of)))
            self.show_progress()
        elif text.startswith("[build] ") or text.startswith("[TOoT:R] "):
            said = text.split("] ", 1)[1]
            if not said.startswith(("/", "cmake ", "ninja ")) and " ... " not in said and not said.endswith("..."):
                self.doing.configure(text=said[:120])
        self.write(line)

    def pump(self):
        try:
            for _ in range(400):
                item = self.lines.get_nowait()
                if isinstance(item, tuple):
                    self.finished(*item)
                else:
                    self.read(item)
        except queue.Empty:
            pass
        if self.process and self.started:
            self.show_progress()
        self.after(100, self.pump)

    def start(self):
        nup = os.path.abspath(os.path.expanduser(self.nup.get().strip()))
        self.target = os.path.abspath(os.path.expanduser(self.folder.get().strip() or DEFAULT_DIR))
        deps = bool(self.why_not_bundled) and self.deps.get()
        self.progress_page()
        self.started = self.step_started = time.time()
        threading.Thread(target=self.work, args=(nup, self.target, deps), daemon=True).start()

    def run(self, cmd, env=None):
        self.lines.put("$ " + " ".join(cmd) + "\n")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, errors="replace")
        for line in self.process.stdout:
            self.lines.put(line)
        code = self.process.wait()
        self.process = None
        return code

    def work(self, nup, target, deps):
        try:
            if deps:
                self.lines.put("@step packages\n")
                manager = next((m for m in DEPS if shutil.which(m)), None)
                if not manager:
                    self.lines.put((False, "This distribution's package manager is not known. Install clang, lld, cmake, ninja, python3, pkg-config "
                                           "and the development files of SDL3, opus, FFmpeg, Vulkan, fmt, zstd and lz4, then install again "
                                           "without \"Install them now\"."))
                    return
                elevate = ["pkexec"] if shutil.which("pkexec") and os.geteuid() != 0 else []
                if not elevate and os.geteuid() != 0:
                    self.lines.put((False, "pkexec is missing: install the build tools yourself, then install again without \"Install them now\"."))
                    return
                if self.run(elevate + DEPS[manager]) != 0:
                    self.lines.put((False, "The build tools were not installed (cancelled, or the package manager failed)."))
                    return
            env = dict(os.environ)
            env.pop("OOT_RECOMP_HOME", None)
            for k in ("PYTHONHOME", "PYTHONPATH"):  # (the AppImage's own Python must not leak into another one)
                env.pop(k, None)
            env["PYTHONUNBUFFERED"] = "1"
            # (the installer's own Python runs the launcher when the system has none: a Steam Deck has one, but nothing else is assumed)
            own = os.environ.get("TOOTR_PYTHON") or sys.executable
            python = own if not self.why_not_bundled else (shutil.which("python3") or own)
            cmd = [python, os.path.join(SOURCE, "installer", "launcher.py"), "install", "--source", SOURCE, "--nup", nup, "--dir", target]
            if not self.why_not_bundled:
                cmd += ["--toolchain", TOOLCHAIN]
            cmd += ["--dlss" if self.dlss.get() else "--no-dlss"]
            code = self.run(cmd, env=env)
            self.lines.put((code == 0, "" if code == 0 else "The installation did not finish."))
        except Exception as e:
            self.lines.put((False, f"The installation failed: {e}"))

    # ---- page 3: done, or what went wrong -----------------------------------------------------------
    def finished(self, ok, message):
        if ok:
            self.begin("done")
            self.done_page()
            return
        self.heading.configure(text="The installation stopped", fg=BAD)
        for key, (mark, name, _) in self.rows.items():
            if self.state_of.get(key) == "running":
                mark.configure(text="✗", fg=BAD)
        last = self.log.get("end-12l", "end").strip().splitlines()
        reason = next((l for l in reversed(last) if "error" in l.lower() or "does not match" in l or "not found" in l), "")
        self.doing.configure(text=(message + ("  " + reason.replace("[build] error: ", "") if reason else "")).strip(), fg=BAD, wraplength=820, justify="left")
        self.remaining.configure(text="")
        if not self.details_shown:
            self.toggle_details()
        self.cancel.configure(text="Close")
        back = ttk.Button(self.cancel.master, text="Back", command=self.setup_page)
        back.pack(side="right", padx=(0, 10))

    def done_page(self):
        self.clear()
        b = self.body
        self.label(b, "✓  TOoT:R is installed", GOOD, 17, True).pack(anchor="w", pady=(10, 4))
        self.label(b, "It took %s. Updates are checked for, and built, each time the game starts." % duration(time.time() - self.started), MUTED).pack(anchor="w")
        tips = tk.Frame(b, bg=PANEL); tips.pack(fill="x", pady=22)
        for title, text in (
            ("Start it", "from your application menu (TOoT:R), or run  tootr  in a terminal."),
            ("Settings", "F1 or Insert in the game opens them: graphics, controller, keyboard and mouse. On a controller, press Select and Start together."),
            ("Steam Deck", "in Desktop Mode open Steam, choose Games > Add a Non-Steam Game and tick TOoT:R; it then appears in Gaming Mode."),
            ("Installed in", self.target),
        ):
            row = tk.Frame(tips, bg=PANEL); row.pack(fill="x", padx=16, pady=7)
            self.label(row, title, ACCENT, 10, True, width=12).pack(side="left", anchor="n")
            self.label(row, text, TEXT, wraplength=660).pack(side="left", fill="x")
        bottom = tk.Frame(b, bg=BG); bottom.pack(side="bottom", fill="x")
        ttk.Button(bottom, text="Play", style="Go.TButton", command=self.launch).pack(side="right")
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right", padx=(0, 10))
        if shutil.which("xdg-open"):
            ttk.Button(bottom, text="Open the folder", command=lambda: subprocess.Popen(["xdg-open", self.target], env=self.clean_env())).pack(side="left")

    def clean_env(self):
        env = dict(os.environ)
        for k in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
            env.pop(k, None)
        return env

    def launch(self):
        launcher = os.path.join(os.path.expanduser("~"), ".local", "bin", "tootr")
        subprocess.Popen([launcher], env=self.clean_env(), start_new_session=True)
        self.destroy()

    def close(self):
        if self.process:
            if not messagebox.askyesno("TOoT:R", "Stop the installation?\n\nWhat is built so far is kept: installing again carries on from there."):
                return
            self.process.terminate()
        self.destroy()


if __name__ == "__main__":
    Installer().mainloop()
