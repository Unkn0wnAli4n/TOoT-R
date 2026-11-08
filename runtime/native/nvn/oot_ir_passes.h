// IR passes specific to this game, ported from Hookshot (github.com/NotRexed/Hookshot,
// gpu/vk/eden/shader_recompiler/ir_opt/oot_visual_workaround.h, GPL-3.0-or-later), whose fog
// matcher in turn comes from Orchid (MythosOfs/Orchid). Included only by shader_compiler.cpp.
#pragma once
#include <cstdlib>
#include <utility>
#include <vector>
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace nvshader::passes {
using namespace Shader;

// No NaN or infinity into a float colour target. One such pixel in the scene target spreads
// through the bloom's chain of averages to the whole picture: a one-frame white flash.
// float_targets: bit i set when colour target i is a float target.
inline size_t GuardFragmentOutputs(IR::Program& program, u32 float_targets) {
    if (program.stage != Stage::Fragment) return 0;
    std::vector<std::pair<IR::Block*, IR::Inst*>> writes;
    for (IR::Block* block : program.blocks)
        for (IR::Inst& inst : block->Instructions())
            if (inst.GetOpcode() == IR::Opcode::SetFragColor && inst.Arg(0).IsImmediate() &&
                (float_targets >> inst.Arg(0).U32() & 1) != 0 && !inst.Arg(2).IsImmediate())
                writes.emplace_back(block, &inst);
    for (const auto& [block, inst] : writes) {
        IR::IREmitter ir{*block, block->Instructions().iterator_to(*inst)};
        const IR::F32 value{inst->Arg(2)};
        // (the largest finite value of the half and 11/10-bit float targets: 1e30 became infinity in them, and an
        // infinite pixel stays in the temporal history and the bloom: a white figure that never fades)
        const IR::F32 limited{ir.FPClamp(value, ir.Imm32(-65000.0f), ir.Imm32(65000.0f))};
        const IR::F32 safe{ir.Select(ir.FPIsNan(value), ir.Imm32(0.0f), limited)};
        inst->SetArg(2, safe);
    }
    return writes.size();
}

// Fog workaround (not a fix): the one compute shader adding 2^x and -2^y is the volumetric fog
// integration; its slice distance is scaled before the extinction/scattering accumulation.
inline size_t ScaleFogIntegration(IR::Program& program, f32 scale) {
    if (program.stage != Stage::Compute || scale == 1.0f) return 0;
    const auto has_opcode = [](const IR::Value& value, IR::Opcode opcode) {
        return !value.IsImmediate() && value.InstRecursive()->GetOpcode() == opcode;
    };
    const auto negative_exp = [&](const IR::Value& value) {
        if (has_opcode(value, IR::Opcode::FPNeg32)) return has_opcode(value.InstRecursive()->Arg(0), IR::Opcode::FPExp2);
        if (!has_opcode(value, IR::Opcode::BitCastF32U32)) return false;
        const auto bits = value.InstRecursive()->Arg(0);
        if (!has_opcode(bits, IR::Opcode::BitwiseXor32)) return false;
        const auto* flip = bits.InstRecursive();
        const auto source = flip->Arg(0);
        return flip->Arg(1).IsImmediate() && flip->Arg(1).U32() == 0x80000000U &&
               has_opcode(source, IR::Opcode::BitCastU32F32) && has_opcode(source.InstRecursive()->Arg(0), IR::Opcode::FPExp2);
    };
    std::vector<std::pair<IR::Block*, IR::Inst*>> matches;
    for (IR::Block* block : program.blocks)
        for (IR::Inst& inst : block->Instructions())
            if (inst.GetOpcode() == IR::Opcode::FPAdd32 &&
                ((negative_exp(inst.Arg(0)) && has_opcode(inst.Arg(1), IR::Opcode::FPExp2)) ||
                 (negative_exp(inst.Arg(1)) && has_opcode(inst.Arg(0), IR::Opcode::FPExp2))))
                matches.emplace_back(block, &inst);
    if (matches.size() != 1) return 0;
    for (const auto& [block, inst] : matches) {
        const auto insertion = block->Instructions().iterator_to(*inst);
        const auto original = block->PrependNewInst(insertion, *inst);
        IR::IREmitter ir{*block, insertion};
        inst->ReplaceUsesWith(ir.FPMul(IR::F32{IR::Value{&*original}}, ir.Imm32(scale)));
    }
    return matches.size();
}

// OOT_FOG_SCALE (default 1: off; Hookshot uses 0.08, Orchid 0.25). OOT_NAN_GUARD=0 disables the guard.
inline f32 FogScale() { static const f32 s = std::getenv("OOT_FOG_SCALE") ? (f32)std::atof(std::getenv("OOT_FOG_SCALE")) : 1.0f; return s; }
inline bool NanGuard() { static const bool on = !std::getenv("OOT_NAN_GUARD") || std::atoi(std::getenv("OOT_NAN_GUARD")) != 0; return on; }
} // namespace nvshader::passes
