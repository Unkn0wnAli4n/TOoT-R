// (longshot) REGMAP_nx2: the prefix frontend/ampere/lower_to_maxwell.h puts before each lowered word of a native
// instruction whose GPRs reach the lowering's reserved registers (RegisterMapPrefix). The translator reads it when it
// comes to the next word (Translate, translate.cpp): each run's numbers name IR registers 256 + its target there. The
// prefix itself does nothing.

#include "common/common_types.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {

void TranslatorVisitor::REGMAP_nx2(u64) {}

void TranslatorVisitor::ApplyRegisterMap(u64 insn) {
    // Two runs: numbers at bits 0-7 and 20-27, targets at 8-15 and 28-35, lengths - 1 at 36-37 and 38-39, the run
    // count at 40-41 (bits 16-19 are the predicate, PT)
    const u32 runs = static_cast<u32>((insn >> 40) & 3);
    constexpr u32 number_at[2]{0, 20}, target_at[2]{8, 28};
    for (u32 i = 0; i < runs && i < 2; ++i) {
        const u32 number = static_cast<u32>((insn >> number_at[i]) & 0xff);
        const u32 target = static_cast<u32>((insn >> target_at[i]) & 0xff);
        const u32 length = static_cast<u32>((insn >> (36 + 2 * i)) & 3) + 1;
        for (u32 k = 0; k < length && number + k < 255 && target + k < 255; ++k) {
            register_map[number + k] = static_cast<u16>(256 + target + k);
        }
    }
}

IR::Reg TranslatorVisitor::MapReg(IR::Reg reg) const {
    const size_t index{IR::RegIndex(reg)};
    if (index < register_map.size() && register_map[index] != 0) {
        return static_cast<IR::Reg>(register_map[index]);
    }
    return reg;
}

} // namespace Shader::Maxwell
