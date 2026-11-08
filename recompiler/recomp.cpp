// Static recompiler: lifts every function of a flat AArch64 module image through dynarmic's A64
// frontend and emits C++ that implements it against runtime/ops.h. The output does not depend on
// dynarmic or the emulator; dynarmic is used here only as an instruction decoder.
//
// usage: recomp <image> <load-base-hex> <functions.txt> <out-dir> <symbol-prefix>
//               [--overrides file] [--entries file] [--files N] [--text-size hex]
//   functions.txt  image offsets of function starts (hex, one per line)
//   overrides      "<offset-hex> <name>" lines: functions with native implementations; direct
//                  calls to them are emitted as dispatched calls so the native one runs
//   entries        extra block starts (offsets) inside functions, e.g. jump table targets

#include <algorithm>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt_passes.h"
#include "dynarmic/ir/terminal.h"

using namespace Dynarmic;
namespace Term = IR::Term;

extern thread_local std::jmp_buf* g_assert_escape;

namespace {

constexpr u32 BranchToSelf = 0x14000000; // B #0

struct Module {
    std::vector<u8> image;
    u64 base = 0;
    u64 text_size = 0; // executable extent from the NSO header; 0 keeps legacy image bounds
    std::vector<u64> starts;                 // absolute guest addresses, sorted
    std::unordered_set<u64> start_set;
    std::unordered_map<u64, std::string> overrides; // absolute address -> native symbol
    std::unordered_set<u64> extra_entries;
    std::unordered_set<u64> exported; // exported functions: called through patchable slots
    std::unordered_set<u64> setjmp_targets; // calls returning twice (setjmp): host resume point at the call
    std::string prefix;

    std::optional<u32> Read(u64 vaddr) const {
        if (vaddr < base || vaddr - base + 4 > image.size()) return std::nullopt;
        if (text_size && vaddr - base + 4 > text_size) return std::nullopt;
        u32 w;
        std::memcpy(&w, &image[vaddr - base], 4);
        return w;
    }
};

std::string Hex(u64 v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%llx", static_cast<unsigned long long>(v));
    return buf;
}

std::string FnName(const Module& m, u64 addr) { return m.prefix + "_" + Hex(addr); }
std::string SlotName(const Module& m, u64 addr) { return "slot_" + m.prefix + "_" + Hex(addr); }

// Exported function symbols, from the module's dynamic symbol table.
void ReadExports(Module& m) {
    const auto u32at = [&](u64 off) { u32 v; std::memcpy(&v, &m.image[off], 4); return v; };
    const auto u64at = [&](u64 off) { u64 v; std::memcpy(&v, &m.image[off], 8); return v; };
    const u32 mod0 = u32at(4);
    u64 dyn = mod0 + (s32)u32at(mod0 + 4), symtab = 0, strtab = 0;
    for (;; dyn += 16) {
        const s64 tag = (s64)u64at(dyn);
        if (tag == 0) break;
        if (tag == 6 && !symtab) symtab = u64at(dyn + 8);
        if (tag == 5 && !strtab) strtab = u64at(dyn + 8);
    }
    for (u64 i = 1; symtab + (i + 1) * 24 <= strtab; ++i) {
        const u64 e = symtab + i * 24;
        const u8 info = m.image[e + 4];
        u16 shndx;
        std::memcpy(&shndx, &m.image[e + 6], 2);
        const u64 value = u64at(e + 8);
        if (value && shndx && (info & 0xf) == 2) m.exported.insert(m.base + value);
    }
}

A64::UserConfig MakeConfig() {
    A64::UserConfig conf{};
    conf.dczid_el0 = 4;
    return conf;
}

// Translates one block at `pc`. When `cut` is given, the block ends before any address in it
// (other than pc itself) with a fall-through LinkBlock.
void Lift(const Module& m, u64 pc, const std::unordered_set<u64>* cut, IR::Block& block) {
    const auto read = [&](u64 vaddr) -> std::optional<u32> {
        if (cut && vaddr != pc && cut->count(vaddr)) return BranchToSelf;
        return m.Read(vaddr);
    };
    A64::Translate(block, A64::LocationDescriptor{pc, {}}, read,
                   {.define_unpredictable_behaviour = true, .wall_clock_cntpct = true, .hook_hint_instructions = false});
    static const auto conf = MakeConfig();
    Optimization::Optimize(block, conf, {.vector_multiply_widen = true});
}

bool HasOp(const IR::Block& block, IR::Opcode op) {
    for (const auto& inst : block.instructions)
        if (inst.GetOpcode() == op) return true;
    return false;
}

// The target of a block that ends by setting a constant PC (e.g. after MSR FPCR).
std::optional<u64> ConstantSetPC(const IR::Block& block) {
    std::optional<u64> pc;
    for (const auto& inst : block.instructions)
        if (inst.GetOpcode() == IR::Opcode::A64SetPC)
            pc = inst.GetArg(0).IsImmediate() ? std::optional<u64>(inst.GetArg(0).GetU64()) : std::nullopt;
    return pc;
}

std::optional<u64> PushedReturn(const IR::Block& block) {
    for (const auto& inst : block.instructions)
        if (inst.GetOpcode() == IR::Opcode::PushRSB) return inst.GetArg(0).GetU64();
    return std::nullopt;
}

void CollectLeaf(const Term::LeafTerminal& t, std::vector<u64>& out) {
    std::visit([&](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, Term::LinkBlock> || std::is_same_v<T, Term::LinkBlockFast>)
            out.push_back(A64::LocationDescriptor{x.next}.PC());
    }, t);
}

void CollectTargets(const Term::Terminal& t, std::vector<u64>& out) {
    std::visit([&](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, Term::LeafTerminal>) {
            CollectLeaf(x, out);
        } else if constexpr (std::is_same_v<T, Term::If> || std::is_same_v<T, Term::CheckBit>) {
            CollectLeaf(x.then_, out);
            CollectLeaf(x.else_, out);
        } else if constexpr (std::is_same_v<T, Term::CheckHalt>) {
            CollectLeaf(x.else_, out);
        }
    }, t);
}

// SIMD register numbers overlap the GPR encodings. Decode writes rather than
// treating an instruction's low five bits as a general-purpose destination.
bool WritesGpr(const Module& m, u64 pc, u32 reg) {
    IR::Block block{A64::LocationDescriptor{pc, {}}};
    A64::Translate(block, A64::LocationDescriptor{pc, {}},
        [&](u64 at) -> std::optional<u32> { return at==pc?m.Read(at):BranchToSelf; },
        {.define_unpredictable_behaviour=true,.wall_clock_cntpct=true,.hook_hint_instructions=false});
    for(const auto& inst:block.instructions)
        if((inst.GetOpcode()==IR::Opcode::A64SetX||inst.GetOpcode()==IR::Opcode::A64SetW)&&
           static_cast<u32>(inst.GetArg(0).GetA64RegRef())==reg)return true;
    return false;
}

// Recovers the targets of a compiler-generated jump table ending in `br` at br_pc:
//   adrp/add Xt, table ; ldr{b,h,sb,sh,sw,w} We, [Xt, Xi{, lsl #s}] ; adr Xb, base ;
//   add Xb, Xb, Xe, {lsl,sxt?} #2 ; br Xb
// The entry count comes from a preceding `cmp Wi/Xi, #n` (n + 1 entries); without one, entries
// are read while they stay inside the function.
std::vector<u64> FindJumpTable(const Module& m, u64 br_pc, u64 start, u64 end) {
    std::vector<u64> targets;
    const auto word = [&](u64 pc) -> u32 { auto w = m.Read(pc); return w ? *w : 0; };
    const u32 br = word(br_pc);
    if ((br & 0xfffffc1fu) != 0xd61f0000u) return targets;
    const u32 xb = (br >> 5) & 31;
    // add Xb, Xn, Xe (shifted or extended register)
    u64 pc = br_pc;
    u32 xn = 99, xe = 99, shift = 0;
    for (int i = 0; i < 64 && pc > start; ++i) {
        pc -= 4;
        const u32 w = word(pc);
        if (!WritesGpr(m,pc,xb)) continue;
        if ((w & 0xff200000u) == 0x8b000000u) { xn = (w >> 5) & 31; xe = (w >> 16) & 31; shift = (w >> 10) & 63; break; }
        if ((w & 0xffe00000u) == 0x8b200000u) { xn = (w >> 5) & 31; xe = (w >> 16) & 31; shift = (w >> 10) & 7; break; }
        return targets;
    }
    if (xe == 99) return targets;
    // The entry load, the base (adr into xn, or the table register itself) and the table address.
    u32 entry_size = 0, xt = 99, xi = 99;
    u64 load_pc = 0;
    bool entry_signed = false;
    std::optional<u64> base, table;
    const u64 add_pc = pc;
    for (u64 q = add_pc; q > start && q + 0x2000 > add_pc;) {
        q -= 4;
        const u32 w = word(q);
        const u32 rd = w & 31;
        // opc distinguishes LDR (1) from sign-extending loads (2/3). Do not
        // include it in the class mask: LDRSW tables use signed byte offsets.
        const u32 load_size = w >> 30, load_opc = (w >> 22) & 3;
        const bool scalar_load = (w & 0x3f200c00u) == 0x38200800u && load_opc != 0 &&
                                 !(load_size == 3 && load_opc >= 2) && !(load_size >= 2 && load_opc == 3);
        if (!entry_size && rd == xe && scalar_load) {
            // load (register offset): size in bits 30-31, opc in 22-23
            const u32 size = w >> 30, opc = (w >> 22) & 3;
            entry_size = 1u << size;
            entry_signed = opc >= 2;
            xt = (w >> 5) & 31;
            xi = (w >> 16) & 31;
            load_pc = q;
            continue;
        }
        if (!base && rd == xn && xn != xt && (w & 0x9f000000u) == 0x10000000u) { // adr
            const s64 imm = (s64)(((w >> 5) & 0x7ffff) << 2 | ((w >> 29) & 3)) << 43 >> 43;
            base = q + imm;
            continue;
        }
        if (entry_size && !table && rd == xt) {
            if ((w & 0xff800000u) == 0x91000000u && ((w >> 5) & 31) == xt) { // add xt, xt, #lo12
                const u32 imm = (w >> 10) & 0xfff;
                // find the adrp
                for (u64 r = q; r > start && r + 0x400 > q;) {
                    r -= 4;
                    const u32 a = word(r);
                    if ((a & 0x9f00001fu) == (0x90000000u | xt)) {
                        const s64 page = (s64)(((a >> 5) & 0x7ffff) << 2 | ((a >> 29) & 3)) << 43 >> 31;
                        table = (r & ~0xfffULL) + page + imm;
                        break;
                    }
                }
                break;
            }
            if ((w & 0x9f000000u) == 0x10000000u) { // adr xt, table
                const s64 imm = (s64)(((w >> 5) & 0x7ffff) << 2 | ((w >> 29) & 3)) << 43 >> 43;
                table = q + imm;
                break;
            }
        }
    }
    if (!entry_size) return targets;
    if (!base && xn != xt) return targets;
    // Bound: cmp Wi/Xi, #n (subs zr, xi, #n)
    u64 count = 0;
    for (u64 q = load_pc; q > start && q + 0x200 > load_pc;) {
        q -= 4;
        const u32 w = word(q);
        if ((w & 0x7f80001fu) == 0x7100001fu && ((w >> 5) & 31) == xi) {
            count = ((w >> 10) & 0xfff) << (((w >> 22) & 1) ? 12 : 0);
            count += 1;
            break;
        }
        // Compilers often copy the checked index into a temporary for the load.
        // Follow that copy before looking for a bound, otherwise an unrelated
        // older CMP on the temporary can truncate the table to one entry.
        if ((w & 0x7fe0ffe0u) == 0x2a0003e0u && (w & 31) == xi) {
            xi = (w >> 16) & 31;
            continue;
        }
        // Other arithmetic definitions invalidate a preceding bound. Branches
        // and stores do not define their low-five-bit register operand.
        if (WritesGpr(m,q,xi)) break;
    }
    // A loop predecessor may materialize the table after the dispatch. A
    // backward scan can also pick up an unrelated, older value of that register.
    // Prefer a unique candidate whose entire bounded table stays in the function.
    std::set<u64> candidates;
    for (u64 q = start; q + 8 <= end; q += 4) {
        const u32 a = word(q);
        if ((a & 0x9f00001fu) != (0x90000000u | xt)) continue;
        const s64 page = (s64)(((a >> 5) & 0x7ffff) << 2 | ((a >> 29) & 3)) << 43 >> 31;
        for (u64 r = q + 4; r < end && r <= q + 16; r += 4) {
            const u32 add = word(r);
            if ((add & 0xffc003ffu) == (0x91000000u | (xt << 5) | xt)) {
                candidates.insert((q & ~0xfffULL) + page + ((add >> 10) & 0xfff));
                break;
            }
        }
    }
    const auto entryTarget = [&](u64 candidate, u64 k) -> std::optional<u64> {
        const u64 at = candidate + k * entry_size;
        if (at < m.base || at + entry_size > m.base + m.image.size()) return {};
        s64 e = 0;
        std::memcpy(&e, &m.image[at - m.base], entry_size);
        if (entry_signed) e = (e << (64 - 8 * entry_size)) >> (64 - 8 * entry_size);
        const u64 t = (xn == xt ? candidate : *base) + ((u64)e << shift);
        if (t < start || t >= end || (t & 3)) return {};
        return t;
    };
    if (count && count <= 4096) {
        std::vector<u64> valid;
        if (table) candidates.insert(*table);
        for (u64 candidate : candidates) {
            bool complete = true;
            for (u64 k = 0; k < count; ++k) {
                if (!entryTarget(candidate, k)) { complete = false; break; }
            }
            if (complete) valid.push_back(candidate);
        }
        if (valid.size() == 1) table = valid.front();
    }
    if (!table && candidates.size() == 1) table = *candidates.begin();
    if (!table) return targets;
    const u64 limit = count ? count : 4096;

    for (u64 k = 0; k < limit; ++k) {
        const auto t = entryTarget(*table, k);
        if (!t) {
            if (count) continue;
            break;
        }
        targets.push_back(*t);
    }
    return targets;
}

// Address-taken code: relocated data pointers into the code range (vtables, callback tables) and
// code addresses materialized with ADR / ADRP+ADD outside the current function. Functions reached
// only through pointers often have no unwind entry.
void AddAddressTakenFunctions(Module& m, const std::vector<u64>& sorted) {
    const u64 code_end = m.text_size ? m.base + m.text_size : sorted.back() + 0x10000;
    const auto u32at = [&](u64 off) { u32 v; std::memcpy(&v, &m.image[off], 4); return v; };
    const auto u64at = [&](u64 off) { u64 v; std::memcpy(&v, &m.image[off], 8); return v; };
    const auto plausible = [&](u64 target) {
        if (target < m.base || target >= code_end || (target & 3)) return false;
        const auto w = m.Read(target);
        return w && *w != 0;
    };
    size_t added = 0;
    // RELR-relocated words hold module offsets in the file.
    const u32 mod0 = u32at(4);
    u64 dyn = mod0 + (s32)u32at(mod0 + 4), relr = 0, relr_size = 0;
    for (;; dyn += 16) {
        const s64 tag = (s64)u64at(dyn);
        if (tag == 0) break;
        if (tag == 36) relr = u64at(dyn + 8);
        if (tag == 35) relr_size = u64at(dyn + 8);
    }
    const auto consider = [&](u64 where) {
        const u64 target = m.base + u64at(where);
        if (plausible(target) && m.start_set.insert(target).second) ++added;
    };
    u64 where = 0;
    for (u64 off = 0; off < relr_size; off += 8) {
        u64 entry = u64at(relr + off);
        if ((entry & 1) == 0) {
            where = entry;
            consider(where);
            where += 8;
        } else {
            for (int i = 0; (entry >>= 1) != 0; ++i)
                if (entry & 1) consider(where + i * 8);
            where += 63 * 8;
        }
    }
    // ADR / ADRP+ADD in code.
    for (size_t i = 0; i < sorted.size(); ++i) {
        const u64 end = i + 1 < sorted.size() ? sorted[i + 1] : sorted[i] + 4;
        for (u64 pc = sorted[i]; pc < end; pc += 4) {
            const auto w = m.Read(pc);
            if (!w) break;
            u64 target = 0;
            if ((*w & 0x9f000000u) == 0x10000000u) {
                target = pc + ((s64)(((*w >> 5) & 0x7ffff) << 2 | ((*w >> 29) & 3)) << 43 >> 43);
            } else if ((*w & 0x9f000000u) == 0x90000000u) {
                const auto next = m.Read(pc + 4);
                const u32 rd = *w & 31;
                if (next && (*next & 0xff800000u) == 0x91000000u && ((*next >> 5) & 31) == rd && (*next & 31) == rd)
                    target = (pc & ~0xfffULL) + ((s64)(((*w >> 5) & 0x7ffff) << 2 | ((*w >> 29) & 3)) << 43 >> 31) +
                             ((*next >> 10) & 0xfff);
            }
            if (target && (target < sorted[i] || target >= end) && plausible(target) && m.start_set.insert(target).second)
                ++added;
        }
    }
    std::fprintf(stderr, "%zu address-taken functions added\n", added);
}

struct FunctionPlan {
    u64 start, end;
    std::vector<u64> labels; // sorted
};

// Finds the block starts of one function by following branches from its entry.
FunctionPlan Discover(const Module& m, u64 start, u64 end) {
    std::set<u64> labels{start};
    std::vector<u64> work{start};
    for (u64 e : m.extra_entries)
        if (e > start && e < end && labels.insert(e).second) work.push_back(e);
    while (!work.empty()) {
        const u64 pc = work.back();
        work.pop_back();
        IR::Block block{A64::LocationDescriptor{pc, {}}};
        Lift(m, pc, nullptr, block);
        std::vector<u64> next;
        const auto ret = PushedReturn(block);
        if (ret) {
            next.push_back(*ret); // the call returns here
        } else {
            CollectTargets(block.GetTerminal(), next);
            if (const auto pc = ConstantSetPC(block)) next.push_back(*pc);
            else if (HasOp(block, IR::Opcode::A64SetPC)) {
                const u64 br_pc = A64::LocationDescriptor{block.EndLocation()}.PC() - 4;
                for (u64 t : FindJumpTable(m, br_pc, start, end)) next.push_back(t);
            }
        }
        // A call block's LinkBlock is the callee, not a label of this function.
        for (u64 t : next) {
            if (t >= start && t < end && (t != start || !ret) && labels.insert(t).second) work.push_back(t);
        }
    }
    return {start, end, {labels.begin(), labels.end()}};
}

const char* CType(IR::Type t) {
    switch (t) {
    case IR::Type::U1: return "bool";
    case IR::Type::U8: return "u8";
    case IR::Type::U16: return "u16";
    case IR::Type::U32: return "u32";
    case IR::Type::U64: return "u64";
    case IR::Type::U128: return "V128";
    case IR::Type::NZCVFlags: return "u32";
    case IR::Type::Table: return "VTable";
    default: return nullptr;
    }
}

class Emitter {
public:
    Emitter(const Module& m, std::ostream& os, std::set<std::string>& externs)
        : m(m), os(os), externs(externs) {}

    void Function(const FunctionPlan& plan) {
        labels = std::unordered_set<u64>(plan.labels.begin(), plan.labels.end());
        plan_ = &plan;
        os << "void " << FnName(m, plan.start) << "(Cpu* c) {\n";
        os << "    u64 next_pc; bool check_bit = false; (void)next_pc; (void)check_bit;\n";
        uses_dispatch = false;
        for (u64 label : plan.labels) {
            IR::Block block{A64::LocationDescriptor{label, {}}};
            Lift(m, label, &labels, block);
            Block(block, label);
        }
        if (uses_dispatch) {
            os << "local_dispatch:\n    switch (next_pc) {\n";
            for (u64 l : plan.labels) os << "    case 0x" << Hex(l) << "ULL: goto L_" << Hex(l) << ";\n";
            os << "    default: rt_call(c, next_pc); return;\n    }\n";
        }
        os << "}\n\n";
    }

    std::string Call(u64 target) {
        if (m.setjmp_targets.count(target)) {
            // setjmp returns again when the guest longjmps: the runtime's native longjmp restores
            // the registers saved here and resumes this frame after the call.
            externs.insert("!setjmp");
            return "if (RT_SETJMP(rt_setjmp_frame(c)) == 0) { " + PlainCall(target) + " }";
        }
        return PlainCall(target);
    }
    std::string PlainCall(u64 target) {
        if (m.overrides.count(target)) return "rt_call(c, 0x" + Hex(target) + "ULL);";
        if (m.exported.count(target)) {
            externs.insert("@" + SlotName(m, target));
            return SlotName(m, target) + "(c);";
        }
        if (m.start_set.count(target)) {
            externs.insert(FnName(m, target));
            return FnName(m, target) + "(c);";
        }
        return "rt_call(c, 0x" + Hex(target) + "ULL);";
    }

private:
    const Module& m;
    std::ostream& os;
    std::set<std::string>& externs;
    std::unordered_set<u64> labels;
    const FunctionPlan* plan_ = nullptr;
    std::unordered_map<const IR::Inst*, std::string> names;
    std::unordered_map<const IR::Inst*, std::string> flags;
    unsigned counter = 0;
    std::optional<u64> pushed_return;
    bool has_set_pc = false;
    u64 current_label = 0;
    bool uses_dispatch = false;
    std::optional<u64> constant_pc;

    std::string Arg(const IR::Value& v) {
        if (!v.IsImmediate()) {
            const IR::Inst* inst = v.GetInstRecursive();
            auto it = names.find(inst);
            if (it == names.end()) throw std::runtime_error("use of unnamed value");
            return it->second;
        }
        switch (v.GetType()) {
        case IR::Type::U1: return v.GetU1() ? "true" : "false";
        case IR::Type::U8: return std::to_string(v.GetU8());
        case IR::Type::U16: return std::to_string(v.GetU16());
        case IR::Type::U32: return "0x" + Hex(v.GetU32()) + "u";
        case IR::Type::U64: return "0x" + Hex(v.GetU64()) + "ULL";
        case IR::Type::A64Reg: return std::to_string(static_cast<int>(v.GetA64RegRef()));
        case IR::Type::A64Vec: return std::to_string(static_cast<int>(v.GetA64VecRef()));
        case IR::Type::Cond: return std::to_string(static_cast<int>(v.GetCond()));
        case IR::Type::AccType: return std::to_string(static_cast<int>(v.GetAccType()));
        case IR::Type::Void: return "NoVec{}"; // unused table slot
        default: throw std::runtime_error("unsupported immediate type");
        }
    }

    void Block(IR::Block& block, u64 label) {
        names.clear();
        flags.clear();
        pushed_return = PushedReturn(block);
        constant_pc = ConstantSetPC(block);
        current_label = label;
        has_set_pc = false;
        os << "L_" << Hex(label) << ": {\n";
        for (auto& inst : block.instructions) Inst(inst);
        os << "    ";
        Terminal(block.GetTerminal());
        os << "\n}\n";
    }

    std::string NewName(const IR::Inst& inst) {
        std::string n = "v" + std::to_string(counter++);
        names[&inst] = n;
        return n;
    }

    std::string Args(const IR::Inst& inst, size_t from = 0) {
        std::string s;
        for (size_t i = from; i < inst.NumArgs(); ++i) {
            if (!s.empty()) s += ", ";
            s += Arg(inst.GetArg(i));
        }
        return s;
    }

    void Inst(IR::Inst& inst) {
        using Op = IR::Opcode;
        const Op op = inst.GetOpcode();
        const std::string name_of_op{IR::GetNameOf(op)};
        auto define = [&](const std::string& expr) {
            const char* type = CType(inst.GetType());
            if (!type) throw std::runtime_error("no C type for " + name_of_op);
            os << "    " << type << " " << NewName(inst) << " = " << expr << ";\n";
        };
        switch (op) {
        case Op::Void: case Op::Identity: case Op::PushRSB: return;
        case Op::A64GetX: define("c->x[" + Arg(inst.GetArg(0)) + "]"); return;
        case Op::A64GetW: define("(u32)c->x[" + Arg(inst.GetArg(0)) + "]"); return;
        case Op::A64SetX: os << "    c->x[" << Arg(inst.GetArg(0)) << "] = " << Arg(inst.GetArg(1)) << ";\n"; return;
        case Op::A64SetW: os << "    c->x[" << Arg(inst.GetArg(0)) << "] = (u64)" << Arg(inst.GetArg(1)) << ";\n"; return;
        case Op::A64GetSP: define("c->sp"); return;
        case Op::A64SetSP: os << "    c->sp = " << Arg(inst.GetArg(0)) << ";\n"; return;
        case Op::A64GetS: define("op_GetS(c->v[" + Arg(inst.GetArg(0)) + "])"); return;
        case Op::A64GetD: define("op_GetD(c->v[" + Arg(inst.GetArg(0)) + "])"); return;
        case Op::A64GetQ: define("c->v[" + Arg(inst.GetArg(0)) + "]"); return;
        case Op::A64SetS: os << "    c->v[" << Arg(inst.GetArg(0)) << "] = op_GetS(" << Arg(inst.GetArg(1)) << ");\n"; return;
        case Op::A64SetD: os << "    c->v[" << Arg(inst.GetArg(0)) << "] = op_GetD(" << Arg(inst.GetArg(1)) << ");\n"; return;
        case Op::A64SetQ: os << "    c->v[" << Arg(inst.GetArg(0)) << "] = " << Arg(inst.GetArg(1)) << ";\n"; return;
        case Op::A64GetCFlag: define("((c->nzcv >> 29) & 1) != 0"); return;
        case Op::A64GetNZCVRaw: define("c->nzcv"); return;
        case Op::A64SetNZCVRaw: os << "    c->nzcv = " << Arg(inst.GetArg(0)) << " & 0xf0000000u;\n"; return;
        case Op::A64SetNZCV: os << "    c->nzcv = " << Arg(inst.GetArg(0)) << ";\n"; return;
        case Op::A64GetFPCR: define("c->fpcr"); return;
        case Op::A64SetFPCR: os << "    rt_set_fpcr(c, " << Arg(inst.GetArg(0)) << ");\n"; return;
        case Op::A64GetFPSR: define("c->fpsr"); return;
        case Op::A64SetFPSR: os << "    c->fpsr = " << Arg(inst.GetArg(0)) << ";\n"; return;
        case Op::A64GetTPIDR: define("c->tpidr"); return;
        case Op::A64SetTPIDR: os << "    c->tpidr = " << Arg(inst.GetArg(0)) << ";\n"; return;
        case Op::A64GetTPIDRRO: define("c->tpidrro"); return;
        case Op::A64SetPC: os << "    next_pc = " << Arg(inst.GetArg(0)) << ";\n"; has_set_pc = true; return;
        case Op::A64SetCheckBit: os << "    check_bit = " << Arg(inst.GetArg(0)) << ";\n"; return;
        case Op::A64DataMemoryBarrier: case Op::A64DataSynchronizationBarrier:
            os << "    __atomic_thread_fence(__ATOMIC_SEQ_CST);\n"; return;
        case Op::A64InstructionSynchronizationBarrier: case Op::A64ClearExclusive:
            if (op == Op::A64ClearExclusive) os << "    c->excl_valid = false;\n";
            return;
        case Op::A64ExceptionRaised:
            os << "    rt_exception(c, " << Arg(inst.GetArg(0)) << ", " << Arg(inst.GetArg(1)) << "); return;\n"; return;
        case Op::A64CallSupervisor:
            os << "    c->pc = 0x" << Hex(current_label) << "ULL; rt_svc(c, " << Arg(inst.GetArg(0)) << ");\n"; return;
        case Op::A64GetCNTPCT: define("rt_cntpct()"); return;
        case Op::A64GetCNTFRQ: define("19200000u"); return;
        case Op::A64GetCTR: define("0x8444c004u"); return;
        case Op::A64GetDCZID: define("4u"); return;
        case Op::A64DataCacheOperationRaised: case Op::A64InstructionCacheOperationRaised: return;
        case Op::ConditionalSelect32: case Op::ConditionalSelect64: case Op::ConditionalSelectNZCV:
            define("rt_cond(c->nzcv, " + Arg(inst.GetArg(0)) + ") ? " + Arg(inst.GetArg(1)) + " : " + Arg(inst.GetArg(2)));
            return;
        case Op::GetNZCVFromOp: case Op::GetCarryFromOp: case Op::GetOverflowFromOp: {
            const IR::Inst* parent = inst.GetArg(0).GetInstRecursive();
            const std::string f = flags.at(parent);
            if (op == Op::GetNZCVFromOp) define(f);
            else if (op == Op::GetCarryFromOp) define("((" + f + " >> 29) & 1) != 0");
            else define("((" + f + " >> 28) & 1) != 0");
            return;
        }
        default: break;
        }
        std::string fn = name_of_op;
        if (fn.rfind("A64", 0) == 0) fn = fn.substr(3);
        const bool memory = fn.rfind("ReadMemory", 0) == 0 || fn.rfind("WriteMemory", 0) == 0;
        const bool exclusive = fn.rfind("Exclusive", 0) == 0;
        std::string call;
        if (memory) {
            call = "op_" + fn + "(" + Args(inst, 1) + ")"; // drop the pc argument
        } else if (exclusive) {
            call = "op_" + fn + "(c, " + Args(inst, 1) + ")";
        } else {
            call = "op_" + fn + "(" + Args(inst) + ")";
        }
        const bool wants_flags = inst.GetAssociatedPseudoOperation(Op::GetNZCVFromOp) ||
                                 inst.GetAssociatedPseudoOperation(Op::GetCarryFromOp) ||
                                 inst.GetAssociatedPseudoOperation(Op::GetOverflowFromOp);
        if (inst.GetType() == IR::Type::Void) {
            os << "    " << call << ";\n";
            return;
        }
        define(call);
        if (wants_flags) {
            const std::string f = "f" + std::to_string(counter++);
            flags[&inst] = f;
            os << "    u32 " << f << " = flags_" << fn << "(" << Args(inst) << ");\n";
        }
    }

    void Goto(u64 target) {
        if (labels.count(target)) {
            os << "goto L_" << Hex(target) << ";";
        } else {
            os << Emitter::Call(target) << " return;";
        }
    }

    void Leaf(const Term::LeafTerminal& t) {
        std::visit([&](const auto& x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, Term::LinkBlock> || std::is_same_v<T, Term::LinkBlockFast>) {
                const u64 target = A64::LocationDescriptor{x.next}.PC();
                if (pushed_return) {
                    os << Emitter::Call(target) << " ";
                    Goto(*pushed_return);
                } else {
                    Goto(target);
                }
            } else if constexpr (std::is_same_v<T, Term::PopRSBHint>) {
                // SVC lifts as PushRSB(next) + PopRSBHint: execution continues after it.
                if (pushed_return) Goto(*pushed_return);
                else os << "return;";
            } else if constexpr (std::is_same_v<T, Term::FastDispatchHint> || std::is_same_v<T, Term::ReturnToDispatch>) {
                if (!has_set_pc) {
                    os << "rt_unreachable(c, 0x" << Hex(plan_->start) << "ULL); return;";
                } else if (constant_pc && !pushed_return) {
                    Goto(*constant_pc);
                } else if (pushed_return) {
                    os << "rt_call(c, next_pc); ";
                    Goto(*pushed_return);
                } else {
                    // Indirect jump: a local label (jump table) or a tail call, resolved by the
                    // function's shared dispatch switch.
                    os << "goto local_dispatch;";
                    uses_dispatch = true;
                }
            } else {
                os << "rt_unreachable(c, 0x" << Hex(plan_->start) << "ULL); return;";
            }
        }, t);
    }

    void Terminal(const Term::Terminal& t) {
        std::visit([&](const auto& x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, Term::LeafTerminal>) {
                Leaf(x);
            } else if constexpr (std::is_same_v<T, Term::If>) {
                os << "if (rt_cond(c->nzcv, " << static_cast<int>(x.if_) << ")) { ";
                Leaf(x.then_);
                os << " } else { ";
                Leaf(x.else_);
                os << " }";
            } else if constexpr (std::is_same_v<T, Term::CheckBit>) {
                os << "if (check_bit) { ";
                Leaf(x.then_);
                os << " } else { ";
                Leaf(x.else_);
                os << " }";
            } else if constexpr (std::is_same_v<T, Term::CheckHalt>) {
                Leaf(x.else_);
            } else {
                os << "rt_unreachable(c, 0x" << Hex(plan_->start) << "ULL); return;";
            }
        }, t);
    }
};

// Writes only when the content changed, so unchanged generated files are not rebuilt.
void WriteIfChanged(const std::string& path, const std::string& content) {
    {
        std::ifstream f(path, std::ios::binary);
        if (f) {
            std::string old((std::istreambuf_iterator<char>(f)), {});
            if (old == content) return;
        }
    }
    std::ofstream(path, std::ios::binary) << content;
}

std::vector<u8> ReadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(f), {}};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: recomp <image> <base> <functions> <out-dir> <prefix> [--overrides f] [--setjmp f] [--entries f] [--files N] [--text-size hex]\n");
        return 2;
    }
    Module m;
    m.image = ReadFile(argv[1]);
    m.base = std::stoull(argv[2], nullptr, 16);
    const std::string out_dir = argv[4];
    m.prefix = argv[5];
    size_t file_count = 256;
    for (int i = 6; i + 1 < argc; i += 2) {
        const std::string opt = argv[i];
        std::ifstream f(argv[i + 1]);
        if (opt == "--overrides") {
            std::string line;
            while (std::getline(f, line)) {
                std::istringstream ss(line);
                std::string off, sym;
                if (ss >> off >> sym) m.overrides[m.base + std::stoull(off, nullptr, 16)] = sym;
            }
        } else if (opt == "--setjmp") {
            u64 v;
            while (f >> std::hex >> v) m.setjmp_targets.insert(m.base + v);
        } else if (opt == "--entries") {
            u64 v;
            while (f >> std::hex >> v) m.extra_entries.insert(m.base + v);
        } else if (opt == "--files") {
            file_count = std::stoul(argv[i + 1]);
        } else if (opt == "--text-size") {
            m.text_size = std::stoull(argv[i + 1], nullptr, 16);
            if (!m.text_size || m.text_size > m.image.size()) throw std::runtime_error("invalid executable extent");
        }
    }
    {
        std::ifstream f(argv[3]);
        u64 v;
        while (f >> std::hex >> v)
            if (!m.text_size || v < m.text_size) m.start_set.insert(m.base + v);
    }
    // Direct call targets (BL) are functions too (e.g. PLT stubs without unwind entries).
    // Repeated until stable: each new start splits a range, which can expose more tail calls.
    for (size_t before = 0; before != m.start_set.size();) {
        before = m.start_set.size();
        std::vector<u64> sorted(m.start_set.begin(), m.start_set.end());
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 0; i < sorted.size(); ++i) {
            const u64 end = i + 1 < sorted.size() ? sorted[i + 1] : sorted[i] + 4;
            for (u64 pc = sorted[i]; pc < end; pc += 4) {
                const auto w = m.Read(pc);
                // BL targets, and B targets outside the function (tail calls, PLT stubs).
                const bool bl = w && (*w & 0xfc000000u) == 0x94000000u;
                const bool b = w && (*w & 0xfc000000u) == 0x14000000u;
                if (bl || b) {
                    const s64 imm = static_cast<s64>(static_cast<s32>((*w & 0x03ffffffu) << 6)) >> 4;
                    const u64 target = pc + imm;
                    if (target >= m.base && target < m.base + (m.text_size ? m.text_size : m.image.size()) &&
                        (bl || target < sorted[i] || target >= end))
                        m.start_set.insert(target);
                }
            }
        }
    }
    ReadExports(m);
    for (u64 a : m.exported)
        if (!m.text_size || a - m.base < m.text_size) m.start_set.insert(a);
    {
        std::vector<u64> sorted(m.start_set.begin(), m.start_set.end());
        std::sort(sorted.begin(), sorted.end());
        AddAddressTakenFunctions(m, sorted);
    }
    m.starts.assign(m.start_set.begin(), m.start_set.end());
    std::sort(m.starts.begin(), m.starts.end());
    std::fprintf(stderr, "%zu functions\n", m.starts.size());

    (void)file_count;
    std::ostringstream table;
    table << "#include \"ops.h\"\n";
    for (u64 s : m.starts) table << "void " << FnName(m, s) << "(Cpu*);\n";
    table << "extern const FunctionEntry " << m.prefix << "_functions[] = {\n";
    for (u64 s : m.starts) table << "    {0x" << Hex(s) << "ULL, " << FnName(m, s) << "},\n";
    table << "};\nextern const size_t " << m.prefix << "_function_count = " << m.starts.size() << ";\n";
    // Patchable entry slots of exported functions (the runtime points them at native code).
    std::vector<u64> slots(m.exported.begin(), m.exported.end());
    std::sort(slots.begin(), slots.end());
    for (u64 a : slots)
        if (m.start_set.count(a)) table << "GuestFn " << SlotName(m, a) << " = " << FnName(m, a) << ";\n";
    table << "extern const SlotEntry " << m.prefix << "_slots[] = {\n";
    size_t slot_count = 0;
    for (u64 a : slots)
        if (m.start_set.count(a)) { table << "    {0x" << Hex(a) << "ULL, &" << SlotName(m, a) << "},\n"; ++slot_count; }
    table << "};\nextern const size_t " << m.prefix << "_slot_count = " << slot_count << ";\n";
    WriteIfChanged(out_dir + "/" + m.prefix + "_table.cpp", table.str());

    // Files are split by generated size (compile time grows faster than linearly with file and
    // function size); very large functions get their own "_big" file, built with less optimization.
    constexpr size_t FileBudget = 4u << 20, BigFunction = 1u << 20;
    size_t failures = 0, file_index = 0, big_index = 0;
    std::set<std::string> written;
    std::ostringstream body;
    std::set<std::string> externs;
    const auto flush = [&](const std::string& name, std::ostringstream& text, std::set<std::string>& ext) {
        std::ostringstream out;
        out << "#include \"ops.h\"\n";
        for (const auto& e : ext) {
            if (e == "!setjmp") out << "void** rt_setjmp_frame(Cpu*);\n";
            else if (e[0] == '@') out << "extern GuestFn " << e.substr(1) << ";\n";
            else out << "void " << e << "(Cpu*);\n";
        }
        out << text.str();
        const std::string path = out_dir + "/" + name + ".cpp";
        WriteIfChanged(path, out.str());
        written.insert(name + ".cpp");
        text.str({});
        ext.clear();
    };
    for (size_t i = 0; i < m.starts.size(); ++i) {
        const u64 start = m.starts[i];
        const u64 end = i + 1 < m.starts.size() ? m.starts[i + 1] : m.base + (m.text_size ? m.text_size : m.image.size());
        std::ostringstream fn;
        std::set<std::string> fn_externs;
        std::jmp_buf escape;
        bool failed = false;
        std::string reason;
        if (setjmp(escape) == 0) {
            g_assert_escape = &escape;
            try {
                const auto plan = Discover(m, start, end);
                Emitter e{m, fn, fn_externs};
                e.Function(plan);
            } catch (const std::exception& ex) {
                failed = true;
                reason = ex.what();
            }
        } else {
            failed = true;
            reason = "dynarmic assertion (not code?)";
        }
        g_assert_escape = nullptr;
        if (failed) {
            ++failures;
            std::fprintf(stderr, "function %llx: %s\n", (unsigned long long)start, reason.c_str());
            fn.str({});
            fn_externs.clear();
            fn << "void " << FnName(m, start) << "(Cpu* c) { rt_unreachable(c, 0x" << Hex(start) << "ULL); }\n";
        }
        if (fn.tellp() > (std::streamoff)BigFunction) {
            flush(m.prefix + "_big_" + std::to_string(big_index++), fn, fn_externs);
            continue;
        }
        body << fn.str();
        externs.insert(fn_externs.begin(), fn_externs.end());
        if (body.tellp() > (std::streamoff)FileBudget) flush(m.prefix + "_" + std::to_string(file_index++), body, externs);
    }
    if (body.tellp() > 0) flush(m.prefix + "_" + std::to_string(file_index++), body, externs);
    // Remove files of an earlier, different split.
    for (const auto& entry : std::filesystem::directory_iterator(out_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(m.prefix + "_", 0) == 0 && name != m.prefix + "_table.cpp" && !written.count(name))
            std::filesystem::remove(entry.path());
    }
    std::fprintf(stderr, "%zu files, %zu big functions\n", file_index, big_index);
    std::fprintf(stderr, "done, %zu functions failed\n", failures);
    return failures ? 1 : 0;
}
