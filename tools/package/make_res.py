#!/usr/bin/env python3
"""Writes a Windows .res file (manifest and icon) without a resource compiler: make_res.py ICO OUT.res"""
import struct, sys

MANIFEST = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity type="win32" name="TOoT-R.Installer" version="1.0.0.0"/>
  <dependency><dependentAssembly><assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls"
    version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/></dependentAssembly></dependency>
  <application xmlns="urn:schemas-microsoft-com:asm.v3"><windowsSettings>
    <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
    <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
  </windowsSettings></application>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3"><security><requestedPrivileges>
    <requestedExecutionLevel level="asInvoker" uiAccess="false"/></requestedPrivileges></security></trustInfo>
</assembly>
'''

def entry(rtype, name, data, flags=0x1030):
    header = struct.pack("<II", len(data), 32) + struct.pack("<HH", 0xFFFF, rtype) + struct.pack("<HH", 0xFFFF, name)
    header += struct.pack("<IHHII", 0, flags, 0x0409, 0, 0)
    out = header + data
    return out + b"\0" * (-len(out) % 4)

def main(ico_path, out_path):
    ico = open(ico_path, "rb").read()
    _, _, count = struct.unpack_from("<HHH", ico, 0)
    res = struct.pack("<II", 0, 32) + struct.pack("<HHHH", 0xFFFF, 0, 0xFFFF, 0) + struct.pack("<IHHII", 0, 0, 0, 0, 0)
    group = struct.pack("<HHH", 0, 1, count)
    for i in range(count):
        w, h, colors, _, planes, bits, size, offset = struct.unpack_from("<BBBBHHII", ico, 6 + 16 * i)
        res += entry(3, i + 1, ico[offset:offset + size])               # RT_ICON
        group += struct.pack("<BBBBHHIH", w, h, colors, 0, planes, bits, size, i + 1)
    res += entry(14, 1, group)                                          # RT_GROUP_ICON (IDI 1)
    res += entry(24, 1, MANIFEST)                                       # RT_MANIFEST
    open(out_path, "wb").write(res)

main(sys.argv[1], sys.argv[2])
