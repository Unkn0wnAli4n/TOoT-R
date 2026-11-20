#!/usr/bin/env python3
"""Game-specific patches to the generated C++ (applied by generate.sh; idempotent):

- Render scale (the Graphics screen): the engine picks its dynamic-resolution path when the config bit at
  +0x188 is set (main+0x805bcc), else a fixed path that scales by 1.0 (x9 = 0x3f800000). With
  oot_fixed_resolution set the fixed path is taken, scaled by oot_fixed_scale_bits. The dynamic path runs
  a few frames first (oot_dynres_runs): it fills fields the 3D reads. Found by Hookshot
  (nx2/oot_patches.py, main+0x805BD0 in its layout = 0x800805bd0 here).
"""
import glob, os, sys

BASE = 0x800000000

DECL = ('extern "C" volatile int oot_fixed_resolution;\nextern "C" volatile int oot_dynres_runs;\n'
        'extern "C" volatile unsigned oot_fixed_scale_bits;\nextern "C" volatile unsigned long long oot_dynres_context;\n')
# (the two ways of the dynamic-resolution branch, in the reference version: the dynamic path's block and the fixed one's)
SCALE_DYNAMIC, SCALE_FIXED = 0x805bd4, 0x80640c
SCALE = "c->x[9] = (u64)0x3f800000u;"
SCALE_NEW = "c->x[9] = (u64)oot_fixed_scale_bits;"


def load(addresses):
    """{reference offset in main: offset in this game} from the build's addresses.txt (tools/signatures.py); without a
    file every place is where the reference version has it."""
    if not addresses or not os.path.exists(addresses):
        return None
    found = {}
    for line in open(addresses):
        p = line.split()
        if len(p) == 3 and p[0] == "main":
            found[int(p[1], 16)] = int(p[2], 16)
    return found


def at(found, off):
    """Where the reference version's main+off is in this game, or None."""
    return off if found is None else found.get(off)


def apply(gen, found=None):
    dynamic, fixed = at(found, SCALE_DYNAMIC), at(found, SCALE_FIXED)
    if dynamic is None or fixed is None:
        print("oot_patches: the dynamic resolution branch is not in this game version: Render Resolution stays the game's own")
        return False
    labels = (BASE + dynamic, BASE + fixed)
    branch = "if (check_bit) { goto L_%x; } else { goto L_%x; }" % labels
    patched = "oot_dynres_context = c->x[19];\n    if (check_bit && !oot_fixed_resolution) { oot_dynres_runs++; goto L_%x; } else { goto L_%x; }" % labels
    label = "L_%x: {" % labels[1]
    done = 0
    for path in sorted(glob.glob(os.path.join(gen, "main", "*.cpp"))):
        s = open(path).read()
        if patched in s:
            done += 1
            continue
        if branch not in s:
            continue
        s = s.replace(branch, patched, 1)
        where = s.index(label)
        scale = s.index(SCALE, where)
        if scale - where > 4000:
            print("oot_patches: the fixed path's scale was not where expected: Render Resolution stays the game's own")
            return False
        s = s[:scale] + SCALE_NEW + s[scale + len(SCALE):]
        s = DECL + s
        open(path, "w").write(s)
        done += 1
        print("oot_patches: render scale switch in", os.path.basename(path))
    if done != 1:
        if found is None:
            sys.exit("oot_patches: the dynamic resolution branch was found %d times" % done)
        print("oot_patches: the dynamic resolution branch was found %d times: Render Resolution stays the game's own" % done)
        return False
    return True

# ---- engine hooks (runtime/native/engine_hooks.cpp) ----------------------------------------------------------------
# One call of oot_site(c, id) at the start of a block, of oot_call(c, target, id) around a call, of oot_fn(c, id) at a
# function's head (true: the function returns at once) or of oot_site(c, id) before each of a function's returns. The
# blocks are made to start where a hook goes by data/v131072/main.entries (the instruction after the one named here).
# What each does is in engine_hooks.cpp, so a hook changes without regenerating the game. The sites are Longshot's
# (nx2/oot_patches.py, oot_wide.py, oot_shadow.py, oot_fov.py; the widescreen ones from NX2navi's list, credited there
# to Ivan0xFF, MIT), written for this recompiler's output; every site's instruction word was checked against this
# game version (main.entries is per version).
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
    # Above 60 frames a second the engine's frame step is a part of a frame. Counters the game cuts to whole frames
    # (the instruction after each cut; the result register and the counter are in engine_hooks.cpp's table):
    (0xa68304, 70),   # the pick-up indicator's count
    (0x475fc, 71),    # a water surface's ripple steps
    (0x3cdfc4, 72),   # the time of day's seconds
    (0x3293414, 73), (0x3296b0c, 74), (0x32926cc, 75),  # the Scarecrow's Song: playback, recording, the last note
    (0x632d04, 76),   # the floor web's simulation steps
    (0x38aebf8, 77), (0x38afd24, 78),                   # the physics solver's steps (in w1; s0 the float cut, s1 the steps of a frame)
    (0x415338, 79),                                     # the sound job's step in s0
    (0x335344c, 80),                                    # the music manager's step and seconds in s8, s9
    (0x3353684, 81), (0x33537c4, 82), (0x3353954, 83), (0x3353a50, 84),  # its reads of the battle manager's change flags (w8, at x26+2200)
    (0x261eb20, 85),                                    # the steering update's rate bounds (s1, s0; the step in s8)
    (0x3b5c7cc, 86),                                    # the movie player's update count (w8, the player in x20)
    # Link's life (x19 the actor, w11 the life before, w12 after, at x13; w9 the hit at the first)
    (0x49dcf8, 87), (0x49ded0, 88),
    # a character controller's forward speed in s0 (x0 the controller): a frame's at the first two, a second's at the third
    (0x935428, 89), (0x9354e0, 90), (0x944bf4, 91),
    # flight: the player's movement (x19 the actor) before its ground test, its pose matrix (x25), the controller's
    # warp (x0, the matrix in x1); the view's loop (x20 the view)
    (0x2c2172c, 95), (0x2c21778, 96), (0x6b25c4, 97), (0x4c5068, 98),
    (0x652354, 100),                                    # the scene buffer's format in w9 (26 R11G11B10F, 44 RGBA16F)
    # The picture's size (the Output Resolution row; 2560 x 1440 as shipped): the 3D's largest size in w24, w25; its
    # constants 1920.0 and 1080.0 in w8; the render targets' pool size at x19+1464; a module heap's use in x0 (its budget
    # at x21 + x22 * 8); the display-size object's set-up (its config in x20: the docked size at +8, +12)
    (0xe83000, 101), (0xe83040, 102), (0xe83048, 103), (0x36aea54, 104), (0x3b5b4e0, 105), (0x1087b90, 106),
    # above 60 frames a second: an animation clock's two advances (x20 the entry; the step in s8, the speed in s1 / s0) and
    # the drawn time's store (s0, at x22); the two copies of the driven-body path (Link's ledge pull-up; x20 the controller)
    (0x2afe04, 111), (0x2b01d0, 112), (0x924e50, 113), (0x4d51e4, 114), (0x4d5e9c, 115),
)
CALL_SITES = ((0x436910, 30), (0x436da4, 31), (0x436f1c, 32), (0x437098, 33), (0xa7ca9c, 34),
              (0x3b5b440, 35), (0xb523ac, 36))  # the Graphics module heap being made (its size in x0), and the heap made of its rest  # (the call's return address, id)
RETURN_SITES = ((0x623f10, 40), (0x623b60, 41), (0x2e821f0, 42), (0x2e82220, 43))  # camera turn speeds, returned in s0
HEAD_SITES = ((0x624c80, 50), (0xf5c80, 51), (0xeb380, 52), (0xd7a820, 53), (0x3b3bfb0, 54), (0xdc6cb0, 55),
              (0x150b670, 57), (0x150aec0, 58), (0x1660c60, 59),  # text: a pane's draw, its layout, a glyph
              (0x2cd660, 60), (0x2d1380, 61),                     # a menu screen: a pane taking the focus, its update
              (0x539df0, 92),                                     # the character controller's forward speed setter (s0, a second's)
              (0x415300, 93), (0x3353400, 94),                    # the sound manager's job (its phase in w2), the music manager's update
              (0xba72d0, 99))                                     # a save request (the reason in w1, the mode in w2)
# (a block's start, where to go on when oot_fn says so, id): the player camera's turn back behind Link as he walks
# (state 3's movement path; the join is where the camera's other yaw paths meet)
JUMP_SITES = ((0x627210, 0x625ad8, 56),
              # above 60 frames a second: the floor carry's tenth (1/dt in s0) in its three update modules, back to the
              # full carry; a horse's slow-motion jump blend (its step in s10), back to the normal one
              (0x13a39c, 0x13a074, 107), (0x13dad4, 0x13d714, 108), (0x389af24, 0x389ac30, 109), (0x2c3ae24, 0x2c3ae48, 110),
              # the texture data manager's compaction request (+0x308 = 1), left out until a second has passed
              (0x77727c, 0x777288, 116))

# What each hook is for, as the settings window names it (a game version without a hook's site: that is off).
FEATURES = (
    ((1, 2), "Render Resolution"), ((3, 3), "Shadow Quality"), ((4, 4), "Field of View"), ((5, 5), "Text Speed and Text Size"),
    ((6, 16), "Aspect Ratio and Output Resolution"), ((17, 22), "HUD Placement"), ((30, 33), "the retry of a full model heap"),
    ((34, 34), "the Voice Language kept after a restart"), ((35, 36), "Output Resolution"), ((40, 43), "Camera Speed"),
    ((50, 50), "the direct mouse camera"), ((51, 52), "faster loading (the host's zstd)"), ((53, 55), "Skip Opening"),
    ((56, 56), "the camera's auto-centring switch"), ((57, 59), "text options"), ((60, 61), "the mouse in the game's menus"),
    ((62, 64), "DLSS and FSR (their camera jitter)"), ((70, 86), "correct game logic above 60 frames a second"),
    ((87, 88), "Damage Taken and Infinite Health"), ((89, 92), "Movement Speed"), ((93, 94), "sound above 60 frames a second"),
    ((95, 98), "the Cheats tab's tools (teleport, flight, free camera, ...)"), ((99, 99), "the Autosave switch"),
    ((101, 106), "Output Resolution"), ((107, 115), "correct game logic above 60 frames a second"),
    ((116, 116), "texture heap repacking once a second"),
)


def features_of(ids):
    out = []
    for (low, high), name in FEATURES:
        if any(low <= i <= high for i in ids) and name not in out:
            out.append(name)
    return out


def block_starts(addresses):
    """The offsets in this game's main where a block must start for the hooks (the recompiler's --entries)."""
    found = load(addresses)
    out = [at(found, off) for off, _ in BLOCK_SITES]
    for off, target, _ in JUMP_SITES:
        out += [at(found, off), at(found, target)]
    out += [at(found, SCALE_DYNAMIC), at(found, SCALE_FIXED)]
    return sorted(a for a in out if a is not None)


def hooks(gen, found=None):
    """The hooks put into the generated code, each where its site is in this game (a site this game version does not
    have is left out, with what it was for)."""
    import re
    block = [(at(found, off), ident) for off, ident in BLOCK_SITES]
    call = [(at(found, off), ident) for off, ident in CALL_SITES]
    ret = [(at(found, off), ident) for off, ident in RETURN_SITES]
    head = [(at(found, off), ident) for off, ident in HEAD_SITES]
    jump = [(at(found, off), at(found, target), ident) for off, target, ident in JUMP_SITES]
    absent = [ident for off, ident in block + call + ret + head if off is None] + [ident for off, target, ident in jump if off is None or target is None]
    block = [x for x in block if x[0] is not None]; call = [x for x in call if x[0] is not None]
    ret = [x for x in ret if x[0] is not None]; head = [x for x in head if x[0] is not None]
    jump = [x for x in jump if x[0] is not None and x[1] is not None]
    want = len(block) + len(call) + len(ret) + len(head) + len(jump)
    done = 0
    placed = set()
    for path in sorted(glob.glob(os.path.join(gen, "main", "*.cpp"))):
        s = open(path).read()
        if "oot_site(c," in s or "oot_call(c," in s or "oot_fn(c," in s:
            sys.exit("oot_patches: %s is hooked already" % os.path.basename(path))
        n = 0
        for off, ident in block:
            label = "L_%x: {\n" % (BASE + off)
            if s.count(label) == 1:
                s = s.replace(label, label + "    oot_site(c, %d);\n" % ident, 1); n += 1; placed.add(ident)
        for off, target, ident in jump:
            label = "L_%x: {\n" % (BASE + off)
            if s.count(label) == 1 and s.count("L_%x: {\n" % (BASE + target)) == 1:
                s = s.replace(label, label + "    if (oot_fn(c, %d)) goto L_%x;\n" % (ident, BASE + target), 1); n += 1; placed.add(ident)
        for off, ident in call:
            ret_at = BASE + off
            pattern = re.compile(r"(    c->x\[30\] = 0x%xULL;\n(?:    next_pc = [^\n]+;\n)?    )(?:rt_call\(c, ([^;]+)\)|main_([0-9a-f]+)\(c\)); goto L_%x;" % (ret_at, ret_at))
            hits = pattern.findall(s)
            if len(hits) == 1:
                s = pattern.sub(lambda m: "%soot_call(c, %s, %d); goto L_%x;" % (m.group(1), m.group(2) if m.group(2) else "0x%sULL" % m.group(3), ident, ret_at), s, 1); n += 1; placed.add(ident)
        for off, ident in ret + head:
            top = "void main_%x(Cpu* c) {\n" % (BASE + off)
            where = s.find(top)
            if where < 0:
                continue
            if (off, ident) in head:
                s = s[:where + len(top)] + "    if (oot_fn(c, %d)) return;\n" % ident + s[where + len(top):]; n += 1; placed.add(ident)
            else:
                end = s.find("\nvoid main_", where + len(top)); end = end if end >= 0 else len(s)
                body, k = re.subn(r"\breturn;", "{ oot_site(c, %d); return; }" % ident, s[where:end])
                if k:
                    s = s[:where] + body + s[end:]; n += 1; placed.add(ident)
        if n:
            if s.count('#include "ops.h"\n') != 1:
                sys.exit("oot_patches: %s does not include ops.h once" % os.path.basename(path))
            open(path, "w").write(s.replace('#include "ops.h"\n', '#include "ops.h"\n' + HOOK_DECL, 1))
            done += n
    unplaced = sorted(i for _, i in block + call + ret + head if i not in placed) + sorted(i for _, _, i in jump if i not in placed)
    if found is None and (done != want or absent):
        sys.exit("oot_patches: %d of %d engine hook sites were found" % (done, want))
    print("oot_patches: %d engine hook sites" % done)
    missing = sorted(set(absent + unplaced))
    if missing:
        print("oot_patches: hooks not in this game version: %s" % ", ".join(str(i) for i in missing))
        print("oot_patches: off for this game version: %s" % "; ".join(features_of(missing)))
    return features_of(missing)


if __name__ == "__main__":
    _gen = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "v131072", "gen")
    _found = load(sys.argv[2]) if len(sys.argv) > 2 else None
    _scale = apply(_gen, _found)
    _off = hooks(_gen, _found)
    if len(sys.argv) > 2:  # (for the settings window: what this game version does without)
        with open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[2])), "missing-features.txt"), "w") as _f:
            _f.write("".join(n + "\n" for n in ([] if _scale else ["Render Resolution"]) + [n for n in _off if n != "Render Resolution"]))
