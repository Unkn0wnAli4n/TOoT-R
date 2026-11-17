#!/usr/bin/env python3
"""Game-specific patches to the generated C++ (applied by generate.sh; idempotent):

- Render scale (the Graphics screen): the engine picks its dynamic-resolution path when the config bit at
  +0x188 is set (main+0x805bcc), else a fixed path that scales by 1.0 (x9 = 0x3f800000). With
  oot_fixed_resolution set the fixed path is taken, scaled by oot_fixed_scale_bits. The dynamic path runs
  a few frames first (oot_dynres_runs): it fills fields the 3D reads. Found by Hookshot
  (nx2/oot_patches.py, main+0x805BD0 in its layout = 0x800805bd0 here).
"""
import glob, os, sys

DECL = ('extern "C" volatile int oot_fixed_resolution;\nextern "C" volatile int oot_dynres_runs;\n'
        'extern "C" volatile unsigned oot_fixed_scale_bits;\nextern "C" volatile unsigned long long oot_dynres_context;\n')
BRANCH = "if (check_bit) { goto L_800805bd4; } else { goto L_80080640c; }"
PATCHED = "oot_dynres_context = c->x[19];\n    if (check_bit && !oot_fixed_resolution) { oot_dynres_runs++; goto L_800805bd4; } else { goto L_80080640c; }"
LABEL = "L_80080640c: {"
SCALE = "c->x[9] = (u64)0x3f800000u;"
SCALE_NEW = "c->x[9] = (u64)oot_fixed_scale_bits;"

def apply(gen):
    done = 0
    for path in sorted(glob.glob(os.path.join(gen, "main", "*.cpp"))):
        s = open(path).read()
        if 'oot_dynres_runs++' in s and PATCHED not in s:
            s = s.replace("if (check_bit && !oot_fixed_resolution) { oot_dynres_runs++; goto L_800805bd4; } else { goto L_80080640c; }", PATCHED, 1)
            s = s.replace('extern "C" volatile unsigned oot_fixed_scale_bits;\n', 'extern "C" volatile unsigned oot_fixed_scale_bits;\nextern "C" volatile unsigned long long oot_dynres_context;\n', 1)
            open(path, 'w').write(s)
        if PATCHED in s:
            done += 1
            continue
        if BRANCH not in s:
            continue
        s = s.replace(BRANCH, PATCHED, 1)
        at = s.index(LABEL)
        scale = s.index(SCALE, at)
        if scale - at > 4000:
            sys.exit("oot_patches: the fixed path's scale was not where expected")
        s = s[:scale] + SCALE_NEW + s[scale + len(SCALE):]
        s = DECL + s
        open(path, "w").write(s)
        done += 1
        print("oot_patches: render scale switch in", os.path.basename(path))
    if done != 1:
        sys.exit("oot_patches: the dynamic resolution branch was found %d times" % done)

# ---- engine hooks (runtime/native/engine_hooks.cpp) ----------------------------------------------------------------
# One call of oot_site(c, id) at the start of a block, of oot_call(c, target, id) around a call, of oot_fn(c, id) at a
# function's head (true: the function returns at once) or of oot_site(c, id) before each of a function's returns. The
# blocks are made to start where a hook goes by data/v131072/main.entries (the instruction after the one named here).
# What each does is in engine_hooks.cpp, so a hook changes without regenerating the game. The sites are Longshot's
# (nx2/oot_patches.py, oot_wide.py, oot_shadow.py, oot_fov.py; the widescreen ones from NX2navi's list, credited there
# to Ivan0xFF, MIT), written for this recompiler's output; every site's instruction word was checked against this
# game version (main.entries is per version).
BASE = 0x800000000
HOOK_DECL = ('extern "C" void oot_site(Cpu* c, int id);\nextern "C" void oot_call(Cpu* c, unsigned long long target, int id);\n'
             'extern "C" bool oot_fn(Cpu* c, int id);\n')
BLOCK_SITES = (  # (block start, id)
    (0x31d3efc, 1), (0x31d3f0c, 2),              # render scale: the recompute request, the hard-coded 0.75 / 0.5 flags
    (0x4729f4, 3),                               # the sun's shadow map size, rounded, in w23 (x19 the shadow object)
    (0x4c52fc, 4),                               # the gameplay camera's vertical angle in s0 (x19 the camera)
    (0xafe1bc, 5),                               # a text pane's letter clock step in s0 (x19 the LetterAnimControl)
    (0x1202ee8, 6), (0x31f39fc, 7), (0x31f3a08, 8),                        # the picture's and the scene's size in x8
    (0x4c5338, 9), (0x3485cc4, 10), (0x4f1ec, 11), (0x23b9ac, 12), (0x33b830, 13),  # camera and projection ratios
    (0x77436c, 14), (0x3b5c75c, 15), (0x284828, 16),                       # the 2D canvas, the movie layer, the save's picture
    (0x152a734, 17), (0x152caa4, 18), (0x152cd9c, 19), (0x152cdd8, 20), (0x152ce68, 21), (0x152ac44, 22),  # layout panes
    # the temporal upscalers' camera jitter (Longshot's nx2/oot_jitter.py): after the loads of a view's projection rows as the
    # game fills its uniform buffers (the view in x28, the rows at +0x1a8): rows 0 and 1 in q19, q20; row 0 in q26; row 1 in q6
    (0x6bc70c, 62), (0x6bcf90, 63), (0x6bcfa0, 64),
)
CALL_SITES = ((0x436910, 30), (0x436da4, 31), (0x436f1c, 32), (0x437098, 33), (0xa7ca9c, 34))  # (the call's return address, id)
RETURN_SITES = ((0x623f10, 40), (0x623b60, 41), (0x2e821f0, 42), (0x2e82220, 43))  # camera turn speeds, returned in s0
HEAD_SITES = ((0x624c80, 50), (0xf5c80, 51), (0xeb380, 52), (0xd7a820, 53), (0x3b3bfb0, 54), (0xdc6cb0, 55),
              (0x150b670, 57), (0x150aec0, 58), (0x1660c60, 59),  # text: a pane's draw, its layout, a glyph
              (0x2cd660, 60), (0x2d1380, 61))                     # a menu screen: a pane taking the focus, its update
# (a block's start, where to go on when oot_fn says so, id): the player camera's turn back behind Link as he walks
# (state 3's movement path; the join is where the camera's other yaw paths meet)
JUMP_SITES = ((0x627210, 0x625ad8, 56),)

def hooks(gen):
    import re
    want = len(BLOCK_SITES) + len(CALL_SITES) + len(RETURN_SITES) + len(HEAD_SITES) + len(JUMP_SITES)
    done = 0
    for path in sorted(glob.glob(os.path.join(gen, "main", "*.cpp"))):
        s = open(path).read()
        if "oot_site(c," in s or "oot_call(c," in s or "oot_fn(c," in s:
            sys.exit("oot_patches: %s is hooked already" % os.path.basename(path))
        n = 0
        for off, ident in BLOCK_SITES:
            label = "L_%x: {\n" % (BASE + off)
            if s.count(label) == 1:
                s = s.replace(label, label + "    oot_site(c, %d);\n" % ident, 1); n += 1
        for off, target, ident in JUMP_SITES:
            label = "L_%x: {\n" % (BASE + off)
            if s.count(label) == 1 and s.count("L_%x: {\n" % (BASE + target)) == 1:
                s = s.replace(label, label + "    if (oot_fn(c, %d)) goto L_%x;\n" % (ident, BASE + target), 1); n += 1
        for off, ident in CALL_SITES:
            ret = BASE + off
            pattern = re.compile(r"(    c->x\[30\] = 0x%xULL;\n(?:    next_pc = [^\n]+;\n)?    )(?:rt_call\(c, ([^;]+)\)|main_([0-9a-f]+)\(c\)); goto L_%x;" % (ret, ret))
            found = pattern.findall(s)
            if len(found) == 1:
                s = pattern.sub(lambda m: "%soot_call(c, %s, %d); goto L_%x;" % (m.group(1), m.group(2) if m.group(2) else "0x%sULL" % m.group(3), ident, ret), s, 1); n += 1
        for off, ident in RETURN_SITES + HEAD_SITES:
            head = "void main_%x(Cpu* c) {\n" % (BASE + off)
            at = s.find(head)
            if at < 0:
                continue
            if (off, ident) in HEAD_SITES:
                s = s[:at + len(head)] + "    if (oot_fn(c, %d)) return;\n" % ident + s[at + len(head):]; n += 1
            else:
                end = s.find("\nvoid main_", at + len(head)); end = end if end >= 0 else len(s)
                body, k = re.subn(r"\breturn;", "{ oot_site(c, %d); return; }" % ident, s[at:end])
                if k:
                    s = s[:at] + body + s[end:]; n += 1
        if n:
            if s.count('#include "ops.h"\n') != 1:
                sys.exit("oot_patches: %s does not include ops.h once" % os.path.basename(path))
            open(path, "w").write(s.replace('#include "ops.h"\n', '#include "ops.h"\n' + HOOK_DECL, 1))
            done += n
    if done != want:
        sys.exit("oot_patches: %d of %d engine hook sites were found" % (done, want))
    print("oot_patches: %d engine hook sites" % done)

if __name__ == "__main__":
    _gen = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "v131072", "gen")
    apply(_gen)
    hooks(_gen)
