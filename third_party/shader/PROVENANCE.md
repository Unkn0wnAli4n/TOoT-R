# third_party/shader

- `src/shader_recompiler/`, `src/video_core/textures/astc.*`: Eden/yuzu shader recompiler
  (GPL-3.0-or-later / GPL-2.0-or-later per file) with this project's modifications,
  including its NVN2/Ampere lowering and the fixes made for this project. astc.cpp: CPU path runs
  single-threaded (callers band the work).
- `sirit/`: sirit v1.0.5 (see its LICENSE.txt). `spirv-headers/`: Khronos SPIR-V headers (MIT).
- `src/common/`, `compat/boost/`, `compat/ankerl/`: minimal shims from Hookshot
  (github.com/NotRexed/Hookshot, gpu/vk/compat and gpu/compat, GPL-3.0-or-later); flat_set gained
  a range insert. `compat/fmt_enum_compat.h`: formats enums for newer fmt.
