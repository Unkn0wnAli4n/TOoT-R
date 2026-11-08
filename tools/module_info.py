#!/usr/bin/env python3
"""Writes a flat module image's function starts (from .eh_frame_hdr and exported functions)."""
import struct, sys
img = open(sys.argv[1], 'rb').read()
mod0 = struct.unpack_from('<I', img, 4)[0]
_, dyn, bss_s, bss_e, eh_s, eh_e, obj = struct.unpack_from('<4sIIIIII', img, mod0)
eh = mod0 + eh_s
starts = set()
if eh_s:
    cnt = struct.unpack_from('<I', img, eh + 8)[0]
    for i in range(cnt):
        starts.add(eh + struct.unpack_from('<i', img, eh + 12 + 8 * i)[0])
# Exported function symbols (STT_FUNC with a value) are entry points too.
d = mod0 + dyn
tags = {}
while True:
    tag, val = struct.unpack_from('<qQ', img, d)
    if tag == 0: break
    tags.setdefault(tag, val); d += 16
symtab, strtab = tags[6], tags[5]
i = 1
while symtab + (i + 1) * 24 <= strtab:
    name, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', img, symtab + i * 24)
    if value and shndx and (info & 0xf) == 2: starts.add(value)
    i += 1
open(sys.argv[2], 'w').write(''.join('%x\n' % s for s in sorted(starts)))
print(len(starts), 'functions')
