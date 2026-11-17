# third_party/shader-ls

The shader translator of Longshot (https://git.magicbean.dev/MagicBeanDev/Longshot, GPL-3.0-or-later), itself the
Eden/yuzu shader recompiler with Hookshot's, Lichii's, Fritchii's and Longshot's NVN2/Ampere work. Taken from the
commit in LONGSHOT_COMMIT:

- `src/shader_recompiler`  from `gpu/vk/eden/shader_recompiler`
- `sirit`                  from `gpu/vk/sirit`
- `compat/vk`, `compat/gpu` from `gpu/vk/compat`, `gpu/compat` (the minimal common/boost/fmt shims)
- `PROVENANCE-longshot.md` Longshot's own provenance file for these
- `src/video_core`, `spirv-headers`  as in `third_party/shader`

Changed here: `kGuestPagesBytes` in `backend/spirv/spirv_emit_context.cpp` (40 GiB: this runtime's pools).

Everything on the renderer's side of the translator is this project's own: `runtime/native/nvn/shader_compiler_ls.cpp`
(the bridge) and, in `runtime/native/nvn`, the guest page table (memory.cpp), the pointer table and storage buffer
tables (render.cpp). Longshot's renderer (`gpu/vk/vk_*.cpp`) is not used.
