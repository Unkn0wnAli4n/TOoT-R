#!/usr/bin/env python3
"""setjmp_targets.py <work-dir>: writes <module>.setjmp, the offsets of call targets that return
twice: PLT stubs bound to setjmp-family imports and the module's own setjmp exports."""
import os, struct, sys

NAMES = {b'setjmp', b'_setjmp', b'__setjmp', b'sigsetjmp', b'__sigsetjmp'}
work = sys.argv[1]
for module in ['main', 'sdk', 'subsdk0']:
    img = open(os.path.join(work, module + '.img'), 'rb').read()
    mod0 = struct.unpack_from('<I', img, 4)[0]
    d = mod0 + struct.unpack_from('<i', img, mod0 + 4)[0]
    tags = {}
    while True:
        tag, val = struct.unpack_from('<qQ', img, d)
        if tag == 0: break
        tags.setdefault(tag, val); d += 16
    symtab, strtab = tags[6], tags[5]
    def name(index):
        n = struct.unpack_from('<I', img, symtab + index * 24)[0]
        return img[strtab + n:img.index(b'\0', strtab + n)]
    targets = set()
    # Own exports.
    i = 1
    while symtab + (i + 1) * 24 <= strtab:
        _, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', img, symtab + i * 24)
        if value and shndx and name(i) in NAMES: targets.add(value)
        i += 1
    # PLT stubs: adrp x16, page; ldr x17, [x16, #off]; add x16, x16, #off; br x17 -> GOT slot.
    slots = set()
    if 23 in tags:
        rel, size = tags[23], tags[2]
        for r in range(rel, rel + size, 24):
            offset, info, addend = struct.unpack_from('<QQq', img, r)
            if (info & 0xffffffff) == 1026 and name(info >> 32) in NAMES: slots.add(offset)
    if slots:
        text = len(img)
        size_file = os.path.join(work, module + '.img.text-size')
        if os.path.exists(size_file): text = int(open(size_file).read().strip(), 16)
        for pc in range(0, text - 16, 4):
            w0, w1, w2, w3 = struct.unpack_from('<4I', img, pc)
            if (w0 & 0x9f00001f) != 0x90000010 or w3 != 0xd61f0220: continue
            immlo, immhi = (w0 >> 29) & 3, (w0 >> 5) & 0x7ffff
            page = (pc & ~0xfff) + (((immhi << 2 | immlo) ^ 0x100000) - 0x100000) * 0x1000
            if (w1 & 0xffc003ff) != 0xf9400211: continue
            if page + ((w1 >> 10) & 0xfff) * 8 in slots: targets.add(pc)
    path = os.path.join(work, module + '.setjmp')
    text = ''.join('%x\n' % t for t in sorted(targets))
    if not os.path.exists(path) or open(path).read() != text:
        open(path, 'w').write(text)
    print(module, len(targets), 'setjmp targets')
