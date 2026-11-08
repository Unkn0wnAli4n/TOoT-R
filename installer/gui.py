#!/usr/bin/env python3
"""TOoT:R graphical installer (Linux; packaged as an AppImage by tools/package/appimage.sh).

Asks for your own dump of the game (.nup) and the folder to install into (blank: the default,
~/.local/share/tootr), installs the build tools (through the system's administrator prompt), then
builds the game there with installer/launcher.py and adds the `tootr` launcher and a menu entry.
The source it builds is the one packaged beside it (TOOTR_SOURCE), else this file's repository.
"""
import os, queue, shutil, subprocess, sys, threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.environ.get("TOOTR_SOURCE") or os.path.dirname(HERE)
DEFAULT_DIR = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "tootr")

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


class Installer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TOoT:R Installer")
        self.minsize(720, 520)
        self.lines = queue.Queue()
        self.process = None
        self.done = False
        pad = {"padx": 12, "pady": 6}

        ttk.Label(self, text="TOoT:R", font=("TkDefaultFont", 20, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(self, text="Builds a native PC version of the game from your own copy. Nothing from the game is "
                             "downloaded or uploaded.", wraplength=680).grid(row=1, column=0, columnspan=3, sticky="w", padx=12)

        ttk.Label(self, text="Game file (.nup, update v131072):").grid(row=2, column=0, sticky="w", **pad)
        self.nup = tk.StringVar()
        ttk.Entry(self, textvariable=self.nup).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(self, text="Browse…", command=self.pick_nup).grid(row=2, column=2, **pad)

        ttk.Label(self, text="Install folder:").grid(row=3, column=0, sticky="w", **pad)
        self.folder = tk.StringVar()
        ttk.Entry(self, textvariable=self.folder).grid(row=3, column=1, sticky="ew", **pad)
        ttk.Button(self, text="Browse…", command=self.pick_folder).grid(row=3, column=2, **pad)
        ttk.Label(self, text=f"Leave empty for the default ({DEFAULT_DIR}).", foreground="gray").grid(row=4, column=1, sticky="w", padx=12)

        self.deps = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Install the build tools first (asks for your password)", variable=self.deps).grid(
            row=5, column=0, columnspan=3, sticky="w", **pad)

        self.log = tk.Text(self, height=16, state="disabled", wrap="word", font=("TkFixedFont", 9))
        self.log.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=12)
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.grid(row=7, column=0, columnspan=3, sticky="ew", padx=12, pady=6)

        buttons = ttk.Frame(self)
        buttons.grid(row=8, column=0, columnspan=3, sticky="e", padx=12, pady=(0, 12))
        self.play = ttk.Button(buttons, text="Play", command=self.launch, state="disabled")
        self.play.pack(side="right", padx=(6, 0))
        self.install = ttk.Button(buttons, text="Install", command=self.start)
        self.install.pack(side="right", padx=(6, 0))
        ttk.Button(buttons, text="Close", command=self.close).pack(side="right")

        self.columnconfigure(1, weight=1)
        self.rowconfigure(6, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(100, self.pump)

    # ---- inputs ----
    def pick_nup(self):
        path = filedialog.askopenfilename(title="Your dump of the game", filetypes=[("Switch 2 package", "*.nup"), ("All files", "*")])
        if path:
            self.nup.set(path)

    def pick_folder(self):
        path = filedialog.askdirectory(title="Install into (a TOoT-R folder is made there)")
        if path:
            self.folder.set(os.path.join(path, "TOoT-R"))

    # ---- work ----
    def write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def pump(self):
        try:
            while True:
                item = self.lines.get_nowait()
                if isinstance(item, tuple):
                    self.finished(*item)
                else:
                    self.write(item)
        except queue.Empty:
            pass
        self.after(100, self.pump)

    def start(self):
        nup = self.nup.get().strip()
        if not os.path.isfile(nup):
            messagebox.showerror("TOoT:R", "Choose your .nup file first.")
            return
        folder = self.folder.get().strip() or DEFAULT_DIR
        self.target = os.path.abspath(os.path.expanduser(folder))
        self.install.configure(state="disabled")
        self.progress.start(12)
        threading.Thread(target=self.work, args=(nup, self.target, self.deps.get()), daemon=True).start()

    def run(self, cmd, env=None):
        self.lines.put("$ " + " ".join(cmd) + "\n")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in self.process.stdout:
            self.lines.put(line)
        code = self.process.wait()
        self.process = None
        return code

    def work(self, nup, target, deps):
        try:
            if deps:
                manager = next((m for m in DEPS if shutil.which(m)), None)
                if not manager:
                    self.lines.put("Unknown distribution: install clang, lld, cmake, ninja, python3, pkg-config and the development "
                                   "files of SDL3, opus, FFmpeg, Vulkan, fmt, zstd and lz4, then try again without this option.\n")
                    self.lines.put((False, "The build tools could not be installed."))
                    return
                elevate = ["pkexec"] if shutil.which("pkexec") and os.geteuid() != 0 else []
                if not elevate and os.geteuid() != 0:
                    self.lines.put((False, "pkexec is missing: install the build tools yourself, then try again without that option."))
                    return
                self.lines.put("Installing the build tools...\n")
                if self.run(elevate + DEPS[manager]) != 0:
                    self.lines.put((False, "The build tools were not installed (cancelled, or the package manager failed)."))
                    return
            python = shutil.which("python3") or sys.executable
            env = dict(os.environ)
            env.pop("OOT_RECOMP_HOME", None)
            for k in ("PYTHONHOME", "PYTHONPATH"):  # (the AppImage's own Python must not leak into the system one)
                env.pop(k, None)
            self.lines.put("Building the game (the first build takes 15-60 minutes)...\n")
            code = self.run([python, os.path.join(SOURCE, "installer", "launcher.py"), "install",
                             "--source", SOURCE, "--nup", nup, "--dir", target], env=env)
            self.lines.put((code == 0, "Installed. Start TOoT:R from your application menu, or run tootr." if code == 0 else
                            "The build failed: see the messages above."))
        except Exception as e:
            self.lines.put((False, f"Installation failed: {e}"))

    def finished(self, ok, message):
        self.progress.stop()
        self.write("\n" + message + "\n")
        self.install.configure(state="normal")
        if ok:
            self.done = True
            self.play.configure(state="normal")
            messagebox.showinfo("TOoT:R", message)
        else:
            messagebox.showerror("TOoT:R", message)

    def launch(self):
        launcher = os.path.join(os.path.expanduser("~"), ".local", "bin", "tootr")
        env = dict(os.environ)
        for k in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(k, None)
        subprocess.Popen([launcher], env=env, start_new_session=True)
        self.destroy()

    def close(self):
        if self.process and messagebox.askyesno("TOoT:R", "Stop the installation?"):
            self.process.terminate()
        elif self.process:
            return
        self.destroy()


if __name__ == "__main__":
    Installer().mainloop()
