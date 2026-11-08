#!/usr/bin/env python3
"""overrides.py <runtime-dir> <work-dir>: writes <module>.overrides for every NATIVE("name") whose
name a module exports, so the recompiler routes direct calls to it through the dispatcher."""
import os, re, struct, sys
runtime, work = sys.argv[1], sys.argv[2]
names = set()
for root, _, files in os.walk(runtime):
    for f in files:
        if f.endswith('.cpp'):
            names.update(re.findall(r'NATIVE\("([^"]+)"\)', open(os.path.join(root, f)).read()))
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
    out = []
    i = 1
    while symtab + (i + 1) * 24 <= strtab:
        name, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', img, symtab + i * 24)
        if value and shndx:
            s = img[strtab + name:img.index(b'\0', strtab + name)].decode()
            if s in names: out.append('%x %s' % (value, s))
        i += 1
    path = os.path.join(work, module + '.overrides')
    text = '\n'.join(sorted(out)) + '\n'
    if not os.path.exists(path) or open(path).read() != text:
        open(path, 'w').write(text)
    print(module, len(out), 'overrides')
