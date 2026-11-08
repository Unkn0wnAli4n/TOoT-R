# third_party/dynarmic

The parts of dynarmic (https://github.com/merryhime/dynarmic, 0BSD; as maintained in the Eden
emulator tree, GPL-3.0-or-later per file headers) that the offline recompiler uses as an AArch64
instruction decoder: the A64 frontend and translator, the IR and its optimization passes, the
common floating-point/bit helpers, mcl headers, and the A32 IR emitter pieces the shared
optimization passes reference. No backend/JIT code is included.

Changes for this project:
- `ir/basic_block.{h,cpp}`: instructions are stored in a `std::deque` instead of boost's
  static_vector/stable_vector (stable addresses, no boost dependency).
- `ir/opt_passes.cpp`: `std::vector`/`std::unordered_map` instead of boost small_vector and
  `Common::unordered_map`.
- `compat/common/`: minimal `assert.h`, `logging.h`, `common_types.h` replacing the emulator's
  `common/` library (same assertion semantics: ASSERT soft, UNREACHABLE fatal).
