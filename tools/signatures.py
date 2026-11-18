#!/usr/bin/env python3
"""Places in the game's code found by what the code looks like, not by where it is.

Every hook site (tools/oot_patches.py) and every address the runtime uses (OOT_AT(...) and G(...) in runtime/) is a
place in one game version: update v131072, the version this project was made with. Another update moves them all. A
signature is the machine code around such a place with what moves between builds masked out (branch targets, the
addresses instructions compute); searching another version's code for it finds the same place there.

  signatures.py find <data dir with main.img ...> <data/signatures.txt> <out addresses.txt>
      (the build) every signature searched for in this game's images. Writes "<module> <reference offset> <offset in
      this game>" for each one found exactly once, and prints what was not found or found more than once: those
      hooks and features are left out for this game version.
  signatures.py make <reference data dir> <data/signatures.txt>
      (development, with the reference version's images) writes the signatures of every place the sources name; each
      is grown until it is found exactly once in the reference version.
  signatures.py check <reference data dir> <data/signatures.txt>
      (development) find, on the reference version: every place must come out where it is.

A signature line:  <module> <reference offset> <kind> <index> <words as hex> <masks as hex> [condition]
  kind "code": the place is the instruction <index> of the match;
  kind "data": the instruction <index> of the match is an ADRP, the next one its ADD or LDR/STR: the place is the address
  the pair computes (a global, a table of functions, a pointer's slot).
  An optional last field "s<i>:<hex>": of several matches, the one whose ADRP pair at instruction <i> points at these bytes
  (functions that differ only in the name they ask for).
Pure Python, no packages: the installer's build runs it.
"""
import glob
import os
import re
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASES = {"main": 0x800000000, "subsdk0": 0x810000000, "sdk": 0x820000000}
LONGSHOT_BASE = 0x80020000  # (addresses in notes taken from Longshot: main + this)


def text_of(data, module):
    path = os.path.join(data, module + ".img")
    img = open(path, "rb").read()
    size = len(img)
    try:
        size = min(size, int(open(path + ".text-size").read().strip(), 16))
    except (OSError, ValueError):
        pass
    return img, size & ~3


def word(img, at):
    return struct.unpack_from("<I", img, at)[0]


def mask_of(w, previous):
    """The bits of an instruction that stay the same wherever the code and its data are linked."""
    if (w & 0x7c000000) == 0x14000000:      # B, BL
        return 0xfc000000
    if (w & 0xff000010) == 0x54000000:      # B.cond
        return 0xff00001f
    if (w & 0x7e000000) == 0x34000000:      # CBZ, CBNZ
        return 0xff00001f
    if (w & 0x7e000000) == 0x36000000:      # TBZ, TBNZ
        return 0xfff8001f
    if (w & 0x1f000000) == 0x10000000:      # ADR, ADRP
        return 0x9f00001f
    if (w & 0x3b000000) == 0x18000000:      # LDR (literal)
        return 0xff00001f
    if previous is not None and (previous & 0x9f000000) == 0x90000000:  # after an ADRP: the low part of its address
        rd = previous & 31
        if (w & 0x7f800000) == 0x11000000 and (w >> 5 & 31) == rd:      # ADD (immediate)
            return 0xffc003ff
        if (w & 0x3b000000) == 0x39000000 and (w >> 5 & 31) == rd:      # LDR / STR (unsigned offset)
            return 0xffc003ff
    return 0xffffffff


def pair_target(img, at):
    """The address the ADRP at `at` and the instruction after it compute, or None."""
    w = word(img, at)
    if (w & 0x9f000000) != 0x90000000:
        return None
    imm = ((w >> 5 & 0x7ffff) << 2 | (w >> 29 & 3))
    if imm & 0x100000:
        imm -= 0x200000
    page = (at & ~0xfff) + (imm << 12)
    n = word(img, at + 4)
    rd = w & 31
    if (n & 0x7f800000) == 0x11000000 and (n >> 5 & 31) == rd:
        return page + ((n >> 10 & 0xfff) << (12 if n >> 22 & 1 else 0))
    if (n & 0x3b000000) == 0x39000000 and (n >> 5 & 31) == rd:
        size = n >> 30
        if n >> 26 & 1 and n >> 23 & 1:   # (a 128-bit SIMD load)
            size = 4
        return page + ((n >> 10 & 0xfff) << size)
    return None


def find_all(img, size, words, masks, limit=3):
    """Offsets where the masked words match, at most `limit` of them."""
    count = len(words)
    # (the anchor: the longest run of whole words, searched for as bytes)
    best, run = (0, 0), None
    for i in range(count + 1):
        whole = i < count and masks[i] == 0xffffffff
        if whole and run is None:
            run = i
        if not whole and run is not None:
            if i - run > best[1]:
                best = (run, i - run)
            run = None
    start, length = best
    if not length:
        return []
    length = min(length, 6)
    anchor = b"".join(struct.pack("<I", words[start + i]) for i in range(length))
    out, at = [], 0
    while True:
        at = img.find(anchor, at, size)
        if at < 0:
            break
        base = at - start * 4
        if at % 4 == 0 and base >= 0 and base + count * 4 <= size:
            for i in range(count):
                if (word(img, base + i * 4) ^ words[i]) & masks[i]:
                    break
            else:
                out.append(base)
                if len(out) >= limit:
                    break
        at += 4 if at % 4 == 0 else 1
    return out


def signature_at(img, size, at):
    """(words, masks, index) of a signature found once in img with instruction `index` at `at`, or None."""
    for reach in (4, 6, 8, 12, 16, 24, 32, 48, 64):
        first = max(0, at - reach * 4)
        last = min(size, at + (reach + 1) * 4)
        words = [word(img, a) for a in range(first, last, 4)]
        masks = [mask_of(w, words[i - 1] if i else (word(img, first - 4) if first else None)) for i, w in enumerate(words)]
        if find_all(img, size, words, masks, 2) == [first]:
            return words, masks, (at - first) // 4, ""
    # Found more than once at every size: functions that differ only in an address they compute (the name of what they
    # ask for, say). The text that address points at tells them apart.
    for reach in (8, 16):
        first = max(0, at - reach * 4)
        last = min(size, at + (reach + 1) * 4)
        words = [word(img, a) for a in range(first, last, 4)]
        masks = [mask_of(w, words[i - 1] if i else None) for i, w in enumerate(words)]
        hits = find_all(img, size, words, masks, 20000)
        if first not in hits or len(hits) >= 20000:
            continue
        for i in range(len(words) - 1):
            mine = pair_target(img, first + i * 4)
            if mine is None or mine + 24 > len(img):
                continue
            text = img[mine:mine + 24]
            others = [pair_target(img, h + i * 4) for h in hits if h != first]
            if all(o is not None and o + 24 <= len(img) and img[o:o + 24] != text for o in others):
                return words, masks, (at - first) // 4, "s%d:%s" % (i, text.hex())
    return None


def references(img, size, wanted):
    """{address: [offsets of ADRPs whose pair computes it]} for the addresses wanted."""
    found = {a: [] for a in wanted}
    for at in range(0, size - 4, 4):
        if (img[at + 3] & 0x9f) == 0x90:
            target = pair_target(img, at)
            if target in found:
                found[target].append(at)
    return found


def places():
    """Every place the sources name: {(module, reference offset)} for code, the runtime's addresses for code or data."""
    sites, used = set(), set()
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import oot_patches
    for off, _ in oot_patches.BLOCK_SITES + oot_patches.CALL_SITES + oot_patches.RETURN_SITES + oot_patches.HEAD_SITES:
        sites.add(("main", off))
    for off, target, _ in oot_patches.JUMP_SITES:
        sites.add(("main", off)), sites.add(("main", target))
    sites.add(("main", oot_patches.SCALE_DYNAMIC)), sites.add(("main", oot_patches.SCALE_FIXED))
    for path in glob.glob(os.path.join(ROOT, "runtime", "**", "*.cpp"), recursive=True) + glob.glob(os.path.join(ROOT, "runtime", "**", "*.h"), recursive=True):
        text = open(path).read()
        for module, off in re.findall(r"OOT_AT_IN\((\w+), 0x([0-9a-f]+)\)", text):
            used.add((module, int(off, 16)))
        for off in re.findall(r"OOT_AT\(0x([0-9a-f]+)\)", text):
            used.add(("main", int(off, 16)))
        if os.path.basename(path) == "debug_tools.cpp":
            for a in re.findall(r"\bG\(0x([0-9a-f]+)\)", text):
                used.add(("main", int(a, 16) - LONGSHOT_BASE))
    return sites, used


def make(data, out_path):
    sites, used = places()
    lines, bad = [], []
    for module in ("main", "sdk", "subsdk0"):
        code = sorted(o for m, o in sites if m == module)
        named = sorted(o for m, o in used if m == module)
        if not code and not named:
            continue
        img, size = text_of(data, module)
        in_text = [o for o in named if o < size]
        in_data = [o for o in named if o >= size]
        refs = references(img, size, in_data) if in_data else {}
        for off in sorted(set(code + in_text)):
            sig = signature_at(img, size, off)
            if not sig:
                bad.append("%s %x: no signature is found once" % (module, off))
                continue
            lines.append(("%s %x code %d %s %s %s" % (module, off, sig[2], ",".join("%08x" % w for w in sig[0]), ",".join("%08x" % m for m in sig[1]), sig[3])).rstrip())
        for off in in_data:
            done = 0
            for at in refs.get(off, []):  # (up to three places that compute it: one that changed still leaves two)
                sig = signature_at(img, size, at)
                if sig:
                    lines.append(("%s %x data %d %s %s %s" % (module, off, sig[2], ",".join("%08x" % w for w in sig[0]), ",".join("%08x" % m for m in sig[1]), sig[3])).rstrip())
                    done += 1
                    if done == 3:
                        break
            if not done:
                bad.append("%s %x: no code computes this address (%d places looked at)" % (module, off, len(refs.get(off, []))))
    with open(out_path, "w") as f:
        f.write("# Made by tools/signatures.py make from the reference version (update v131072). See that file.\n" + "\n".join(lines) + "\n")
    print("signatures: %d written for %d places" % (len(lines), len(sites | used)))
    for b in bad:
        print("signatures: " + b)
    return not bad


def read(path):
    for line in open(path):
        p = line.split()
        if len(p) not in (6, 7) or line.startswith("#"):
            continue
        yield p[0], int(p[1], 16), p[2], int(p[3]), [int(w, 16) for w in p[4].split(",")], [int(m, 16) for m in p[5].split(",")], p[6] if len(p) == 7 else ""


def find(data, sig_path, out_path, quiet=False):
    images, found, votes, missing = {}, {}, {}, {}
    for module, off, kind, index, words, masks, condition in read(sig_path):
        if module not in images:
            images[module] = text_of(data, module)
        img, size = images[module]
        hits = find_all(img, size, words, masks, 20000 if condition else 2)
        if condition:  # (s<index>:<bytes>: what the ADRP pair at <index> points at begins with these bytes)
            where, text = condition[1:].split(":"); text = bytes.fromhex(text)
            kept = []
            for h in hits:
                target = pair_target(img, h + int(where) * 4)
                if target is not None and img[target:target + len(text)] == text:
                    kept.append(h)
            hits = kept
        key = (module, off)
        if len(hits) != 1:
            missing.setdefault(key, []).append("not found" if not hits else "found more than once")
            continue
        at = hits[0] + index * 4
        value = at if kind == "code" else pair_target(img, at)
        if value is None:
            missing.setdefault(key, []).append("its instructions changed")
            continue
        votes.setdefault(key, []).append(value)
    for key, values in votes.items():
        best = max(set(values), key=values.count)
        if values.count(best) * 2 > len(values) or len(set(values)) == 1:
            found[key] = best
        else:
            missing.setdefault(key, []).append("its signatures disagree")
    lost = sorted(k for k in missing if k not in found)
    with open(out_path, "w") as f:
        for (module, off), value in sorted(found.items()):
            f.write("%s %x %x\n" % (module, off, value))
    if not quiet:
        moved = sum(1 for (m, o), v in found.items() if o != v)
        print("signatures: %d places found (%d at another address than in the reference version), %d not found" % (len(found), moved, len(lost)))
        for module, off in lost:
            print("signatures: %s %x %s" % (module, off, ", ".join(sorted(set(missing[(module, off)])))))
    return found, lost


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "make" and len(sys.argv) == 4:
        sys.exit(0 if make(sys.argv[2], sys.argv[3]) else 1)
    if mode == "find" and len(sys.argv) == 5:
        find(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(0)
    if mode == "check" and len(sys.argv) == 4:
        got, lost = find(sys.argv[2], sys.argv[3], os.devnull)
        wrong = [(k, v) for k, v in got.items() if k[1] != v]
        for (module, off), v in wrong:
            print("signatures: %s %x came out at %x" % (module, off, v))
        sys.exit(1 if wrong or lost else 0)
    sys.exit(__doc__)
