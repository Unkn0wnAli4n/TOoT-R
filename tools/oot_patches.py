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

if __name__ == "__main__":
    apply(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "v131072", "gen"))
