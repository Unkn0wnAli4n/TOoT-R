// (longshot) HMMA_nx2: the pseudo instruction frontend/ampere/lower_to_maxwell.h emits for a native
// HMMA.16816.F16 when the host has cooperative matrices (Shader::Ampere::native_matrix_multiply).
// D(16x8) = A(16x16) * B(16x8) + C over the 32 lanes, f16 throughout; each lane holds its
// fragments in registers as on the console: A in four (a..a+3), B in two, C and D in two.

#include "common/bit_field.h"
#include "common/common_types.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {

void TranslatorVisitor::HMMA_nx2(u64 insn) {
    union {
        u64 raw;
        BitField<0, 8, IR::Reg> dest;
        BitField<8, 8, IR::Reg> a;
        BitField<20, 8, IR::Reg> b;
        BitField<39, 8, IR::Reg> c;
    } const hmma{insn};

    const IR::Reg a{hmma.a}, b{hmma.b}, c{hmma.c}, dest{hmma.dest};
    const IR::Value a_frag{ir.CompositeConstruct(X(a), X(a + 1), X(a + 2), X(a + 3))};
    const IR::Value b_frag{ir.CompositeConstruct(X(b), X(b + 1))};
    const IR::Value c_frag{c == IR::Reg::RZ ? ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0))
                                            : ir.CompositeConstruct(X(c), X(c + 1))};
    const IR::Value d_frag{ir.MatrixMulAddF16(a_frag, b_frag, c_frag)};
    const IR::U32 d0{ir.CompositeExtract(d_frag, 0)}, d1{ir.CompositeExtract(d_frag, 1)};
    X(dest, d0);
    X(dest + 1, d1);
}

} // namespace Shader::Maxwell
