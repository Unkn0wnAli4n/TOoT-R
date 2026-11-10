// Native Ampere instructions -> the existing compiler's IR -> Vulkan SPIR-V.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "shader_compiler.h"
#include "oot_ir_passes.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/nvn_storage_descriptor.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/trap_guards.h"
#include "shader_recompiler/frontend/maxwell/structured_control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "video_core/textures/astc.h"

namespace nvshader {
namespace {
using Shader::Stage;
Shader::TexturePixelFormat PixelFormat(uint32_t format) {
    using F = Shader::TexturePixelFormat;
    if (format >= 121 && format <= 148) return format >= 135 ? F::A8B8G8R8_SRGB : F::A8B8G8R8_UNORM;
    switch (format) {
    case 1: return F::R8_UNORM; case 2: return F::R8_SNORM; case 3: return F::R8_UINT; case 4: return F::R8_SINT;
    case 5: return F::R16_FLOAT; case 6: return F::R16_UNORM; case 7: return F::R16_SNORM;
    case 8: return F::R16_UINT; case 9: return F::R16_SINT;
    case 10: return F::R32_FLOAT; case 11: return F::R32_UINT; case 12: return F::R32_SINT;
    case 13: return F::R8G8_UNORM; case 14: return F::R8G8_SNORM; case 15: return F::R8G8_UINT; case 16: return F::R8G8_SINT;
    case 17: return F::R16G16_FLOAT; case 18: return F::R16G16_UNORM; case 19: return F::R16G16_SNORM;
    case 20: return F::R16G16_UINT; case 21: return F::R16G16_SINT;
    case 22: return F::R32G32_FLOAT; case 23: return F::R32G32_UINT; case 24: return F::R32G32_SINT;
    case 34: return F::R32G32B32_FLOAT;
    case 37: case 109: return F::A8B8G8R8_UNORM; case 38: case 110: return F::A8B8G8R8_SNORM;
    case 39: case 111: return F::A8B8G8R8_UINT; case 40: case 112: return F::A8B8G8R8_SINT;
    case 41: case 113: return F::R16G16B16A16_FLOAT; case 42: case 114: return F::R16G16B16A16_UNORM;
    case 43: case 115: return F::R16G16B16A16_SNORM; case 44: case 116: return F::R16G16B16A16_UINT;
    case 45: case 117: return F::R16G16B16A16_SINT; case 46: case 118: return F::R32G32B32A32_FLOAT;
    case 47: case 119: return F::R32G32B32A32_UINT; case 48: case 120: return F::R32G32B32A32_SINT;
    case 49: return F::S8_UINT; case 50: return F::D16_UNORM; case 51: return F::X8_D24_UNORM;
    case 52: return F::D32_FLOAT; case 53: return F::D24_UNORM_S8_UINT; case 54: return F::D32_FLOAT_S8_UINT;
    case 55: case 56: return F::A8B8G8R8_SRGB;
    case 57: return F::A4B4G4R4_UNORM; case 58: case 59: return F::A1B5G5R5_UNORM;
    case 60: return F::B5G6R5_UNORM; case 61: return F::A2B10G10R10_UNORM; case 62: return F::A2B10G10R10_UINT;
    case 63: return F::B10G11R11_FLOAT; case 64: return F::E5B9G9R9_FLOAT;
    case 65: case 66: return F::BC1_RGBA_UNORM; case 67: return F::BC2_UNORM; case 68: return F::BC3_UNORM;
    case 69: case 70: return F::BC1_RGBA_SRGB; case 71: return F::BC2_SRGB; case 72: return F::BC3_SRGB;
    case 73: return F::BC4_UNORM; case 74: return F::BC4_SNORM; case 75: return F::BC5_UNORM; case 76: return F::BC5_SNORM;
    case 77: return F::BC7_UNORM; case 78: return F::BC7_SRGB; case 79: return F::BC6H_SFLOAT; case 80: return F::BC6H_UFLOAT;
    case 153: case 154: return F::B8G8R8A8_UNORM; case 155: case 156: return F::B8G8R8A8_SRGB;
    default: throw std::runtime_error("NVN texture format not yet mapped into shader compiler: " + std::to_string(format));
    }
}
bool IntegerFormat(uint32_t f) {
    return f == 3 || f == 4 || f == 8 || f == 9 || f == 11 || f == 12 || f == 15 || f == 16 ||
        f == 20 || f == 21 || f == 23 || f == 24 || f == 39 || f == 40 || f == 44 || f == 45 ||
        f == 47 || f == 48 || f == 49 || f == 62 || f == 111 || f == 112 || f == 116 || f == 117 || f == 119 || f == 120;
}
class Environment final : public Shader::Environment {
public:
    explicit Environment(const Request& r) : request(r) {
        const Stage stages[] = {Stage::VertexB, Stage::Fragment, Stage::Geometry, Stage::TessellationControl,
                                Stage::TessellationEval, Stage::Compute, Stage::Task, Stage::Mesh};
        if (r.stage > 7 || !r.code || !r.control) throw std::runtime_error("invalid shader compile request");
        stage = stages[r.stage]; is_proprietary_driver = true;
        mesh_registers = r.mesh_registers;
        if (stage != Stage::Compute) {
            if (r.code->size() < 128) throw std::runtime_error("missing native shader header");
            std::memcpy(&sph, r.code->data(), sizeof sph);
        }
    }
    uint32_t ReadCbufValue(uint32_t bank, uint32_t offset) override {
        const auto value = request.read_constant(request.context, bank, offset);
        // Bank 2 holds the driver's texture/image handles: the code depends on what they name,
        // not on the handle itself.
        const uint32_t kind = request.classify_constant ? request.classify_constant(request.context, bank, offset) : bank == 2 ? 1 : 0;
        if (kind == 1) {
            const uint32_t described = request.describe_texture ? request.describe_texture(request.context, value) :
                request.texture_type(request.context, value) | request.texture_format(request.context, value) << 16;
            dependencies.push_back({bank, offset, described, DependencyKind::Texture});
        } else if (kind == 0) {
            dependencies.push_back({bank, offset, value});
        }
        return value;
    }
    bool IsCbufStorageBufferPointer(uint32_t bank, uint32_t offset) override {
        const auto read = [&](uint32_t o) { return request.read_constant(request.context, bank, o); };
        const bool available=!request.constant_pointer_available||request.constant_pointer_available(request.context,bank,offset);
        const uint64_t address = available?(read(offset) | (uint64_t(read(offset + 4)) << 32)):0;
        const bool mapped = request.mapped_address(request.context, address);
        dependencies.push_back({bank, offset, mapped, DependencyKind::Mapped});
        return mapped;
    }
    bool IsCbufIndirectStorageBufferPointer(uint32_t bank,uint32_t offset,uint32_t indirect) override {
        const bool available=!request.constant_pointer_available||request.constant_pointer_available(request.context,bank,offset);
        const uint64_t base=available?(request.read_constant(request.context,bank,offset)|
            uint64_t(request.read_constant(request.context,bank,offset+4))<<32):0;
        const uint64_t address=request.read_pointer?request.read_pointer(request.context,base+indirect):0;
        const bool mapped=request.mapped_address(request.context,address);
        dependencies.push_back({bank,offset,mapped,DependencyKind::IndirectMapped,indirect});
        return mapped;
    }
    Shader::TextureType ReadTextureType(uint32_t handle) override {
        return static_cast<Shader::TextureType>(request.texture_type(request.context, handle));
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(uint32_t handle) override {
        return PixelFormat(request.texture_format(request.context, handle));
    }
    bool IsTexturePixelFormatInteger(uint32_t handle) override {
        return IntegerFormat(request.texture_format(request.context, handle));
    }
    uint32_t ReadViewportTransformState() override { return 1; }
    uint32_t TextureBoundBuffer() const override { return texture_bound; }
    // The program's local memory plus its call-return stack area. Translated programs have structured control flow and
    // never touch that area; on AMD every byte of local memory is scratch memory per thread (OoT's programs asked for
    // up to 18 KB, a 320 MB scratch buffer on the Steam Deck). OOT_LOCAL_CRS=0/1 forces it out/in; default: out except on NVIDIA.
    uint32_t LocalMemorySize() const override {
        static const bool crs = [] { const char* e = std::getenv("OOT_LOCAL_CRS"); return !e || std::atoi(e) != 0; }(); // (vk.cpp sets 0 on AMD)
        return (uint32_t)sph.LocalMemorySize() + (crs ? (uint32_t)sph.common3.shader_local_memory_crs_size : 0u);
    }
    // NVN2 compute control: +0x4c register count, +0x58.. workgroup size, +0x64 shared memory bytes.
    uint32_t SharedMemorySize() const override { return Control(0x64); }
    std::array<uint32_t, 3> WorkgroupSize() const override {
        return {std::max(Control(0x58), 1u), std::max(Control(0x5c), 1u), std::max(Control(0x60), 1u)};
    }
    bool HasHLEMacroState() const override { return false; }
    bool HasNativeInstructions() const override { return true; }
    bool IsNativeProgram() const override { return true; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(uint32_t, uint32_t) override { return {}; }
    void Dump(uint64_t, uint64_t) override {}
    uint64_t ReadInstruction(uint32_t address) override {
        if (address % 8 || address / 8 >= code.size()) throw std::runtime_error("shader instruction outside captured code");
        return code[address / 8];
    }
    uint32_t TranslateProgramAddress(uint32_t address) const override {
        const auto target = Shader::Ampere::NativeProgramAddress(code, address);
        if (!target) throw std::runtime_error("native shader branch target not captured");
        return *target;
    }
    void Lower() {
        const size_t header = stage == Stage::Compute ? 0 : 128;
        if ((request.code->size() - header) % 16) throw std::runtime_error("unaligned native code extent");
        std::vector<Shader::Ampere::Instruction> native((request.code->size() - header) / 16);
        std::memcpy(native.data(), request.code->data() + header, native.size() * 16);
        Shader::Ampere::ProgramCaptureState capture;
        bool complete = false;
        for (uint32_t i = 0; i < native.size(); ++i) {
            const auto status = capture.Observe(native[i], i);
            if (status == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) throw std::runtime_error("invalid native shader branch extent");
            if (status == Shader::Ampere::ProgramCaptureState::Result::Complete) { native.resize(i + 1); complete = true; break; }
        }
        if (!complete) throw std::runtime_error("native shader allocation has no complete program");
        std::vector<uint32_t> traps;
        for (uint32_t i = 0; i < native.size(); ++i) if (native[i].Bits(0, 12) == 0x95c) traps.push_back(i * 16);
        if (!traps.empty() && (stage != Stage::Compute || !Shader::Ampere::VerifyTrapGuards(
                std::span{native}.first(traps.back() / 16 + 1), [&](uint32_t bank, uint32_t offset) -> std::optional<uint32_t> {
                    return ReadCbufValue(bank, offset);
                }))) throw std::runtime_error("native compute trap guard is not satisfied");
        const auto tess = stage == Stage::TessellationControl ? Shader::Ampere::TessellationStage::Control :
            stage == Stage::TessellationEval ? Shader::Ampere::TessellationStage::Evaluation : Shader::Ampere::TessellationStage::None;
        const auto lowered = Shader::Ampere::LowerToMaxwell(native, stage == Stage::Geometry, true, traps, tess);
        if (!lowered) throw std::runtime_error("Ampere lowering at " + std::to_string(lowered.error_offset) + ": " + std::string(lowered.error));
        code.resize(sizeof(sph) / 8); std::memcpy(code.data(), &sph, sizeof sph);
        for (auto instruction : lowered.instructions) { if (code.size() % 4 == 0) code.push_back(0); code.push_back(instruction); }
        code.push_back(0); Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
        texture_bound = lowered.required_texture_buffer == ~0u ? 2u : lowered.required_texture_buffer;
    }
    std::vector<Dependency> dependencies;
private:
    uint32_t Control(size_t offset) const {
        uint32_t value = 0;
        if (offset + 4 <= request.control->size()) std::memcpy(&value, request.control->data() + offset, 4);
        return value;
    }
    const Request& request;
    std::vector<uint64_t> code;
    uint32_t texture_bound = 2;
};
}

Result Compile(const Request& request) {
    Environment env(request); env.Lower();
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{sizeof(Shader::ProgramHeader)}};
    Shader::ObjectPool<Shader::IR::Inst> inst_pool;
    Shader::ObjectPool<Shader::IR::Block> block_pool;
    Shader::HostTranslateInfo host{};
    host.support_float64 = host.support_float16 = host.support_int64 = true;
    host.support_conditional_barrier = host.support_viewport_index_layer = true;
    host.min_ssbo_alignment = request.storage_alignment;
    host.ApplyDescriptorLimitPolicy();
    auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
    Shader::RuntimeInfo runtime{};
    for (size_t i = 0; i < 32; ++i) runtime.generic_input_types[i] = static_cast<Shader::AttributeType>(request.input_types[i]);
    for (size_t i = 0; i < 8; ++i) runtime.color_output_types[i] = static_cast<Shader::AttributeType>(request.color_types[i]);
    if (request.have_previous_stores) {
        for (size_t i = 0; i < 512; ++i) runtime.previous_stage_stores.mask[i] = (request.previous_stores[i / 64] >> (i % 64)) & 1;
    } else runtime.previous_stage_stores.mask.set();
    runtime.convert_depth_mode = request.convert_depth; runtime.y_negate = request.negate_y;
    runtime.force_early_z = request.force_early_depth; runtime.dual_source_blend = request.dual_source;
    runtime.input_topology = static_cast<Shader::InputTopology>(request.topology);
    if (request.point_size > 0) runtime.fixed_state_point_size = request.point_size;
    Shader::Maxwell::ConvertLegacyToGeneric(program, runtime);
    if (passes::NanGuard()) {
        uint32_t float_targets = 0;
        for (int i = 0; i < 8; ++i) if (runtime.color_output_types[i] == Shader::AttributeType::Float) float_targets |= 1u << i;
        passes::GuardFragmentOutputs(program, float_targets);
    }
    passes::ScaleFogIntegration(program, passes::FogScale());
    if (program.stage == Stage::Task && passes::MeshNoCull()) {
        const size_t made = passes::MeshletCullingOff(program);
        if (made) Shader::Optimization::DeadCodeEliminationPass(program);
        if (std::getenv("OOT_MESH_NOCULL_TRACE")) std::fprintf(stderr, "[rt] task program: %zu culling tests replaced\n", made);
    }
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010400; profile.unified_descriptor_binding = true;
    profile.min_ssbo_alignment = request.storage_alignment;
    profile.global_address_table = request.global_address_table;
    profile.support_buffer_float32_atomic_add = request.buffer_float32_atomic_add;
    profile.support_int64 = profile.support_int16 = profile.support_int8 = true;
    profile.support_vote = profile.support_shuffle_relative = true;
    profile.support_sampled_image_array_nonuniform_indexing = true;
    profile.support_viewport_index_layer_non_geometry = profile.support_multi_viewport = true;
    profile.max_subgroup_size = 32; profile.max_user_clip_distances = 8;
    // Subgroup operations in every stage including task and mesh (Shader::Stage 7/8; the profile's
    // default 0x7F left them out): without them ballots/votes/shuffles fall back to single-lane
    // results, so mesh programs computed their primitive count per lane (no grass drawn) and task
    // programs launched no mesh workgroups. The host GPU supports subgroups there (checked in vk.cpp).
    profile.supported_subgroup_stages = 0x1FF;
    profile.support_typeless_image_loads = true;
    Shader::Backend::Bindings bindings{}; bindings.unified = request.binding_base;
    Result result; result.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
    result.dependencies = std::move(env.dependencies);
    result.global_loads = program.info.uses_global_memory;
    result.global_stores = program.info.stores_global_memory;
    for (size_t i = 0; i < 512; ++i) if (program.info.stores.mask[i]) result.stores[i / 64] |= 1ULL << (i % 64);
    uint32_t binding = request.binding_base;
    for (const auto& descriptor : program.info.constant_buffer_descriptors) {
        Resource r; r.kind = ResourceKind::UniformBuffer; r.binding = binding;
        r.bank = descriptor.index; r.count = descriptor.count; binding += r.count; result.resources.push_back(std::move(r));
    }
    for (const auto& descriptor : program.info.storage_buffers_descriptors) {
        Resource r; r.kind = ResourceKind::StorageBuffer; r.binding = binding;
        r.bank = descriptor.cbuf_index; r.offset = descriptor.cbuf_offset; r.count = descriptor.count;
        r.indirect_offset=descriptor.indirect_offset;
        r.sized_storage=descriptor.indirect_offset==~0u&&Shader::Backend::UsesNvnStorageDescriptor(program.info,descriptor);
        r.written = descriptor.is_written; binding += r.count; result.resources.push_back(std::move(r));
    }
    const auto append = [&](const auto& descriptors, ResourceKind kind) {
        for (const auto& descriptor : descriptors) {
            Resource r; r.kind = kind; r.binding = binding;
            r.bank = descriptor.cbuf_index; r.offset = descriptor.cbuf_offset; r.count = descriptor.count;
            r.size_shift = descriptor.size_shift;
            if constexpr (requires { descriptor.storage_index; }) {
                r.storage_index = descriptor.storage_index;
                r.storage_offset = descriptor.storage_offset;
                r.storage_stride = descriptor.storage_stride;
            }
            if constexpr (requires { descriptor.type; }) r.texture_type = static_cast<uint32_t>(descriptor.type);
            if constexpr (requires { descriptor.format; }) r.image_format = static_cast<uint32_t>(descriptor.format);
            if constexpr (requires { descriptor.is_depth; }) r.depth = descriptor.is_depth;
            if constexpr (requires { descriptor.is_multisample; }) r.multisample = descriptor.is_multisample;
            if constexpr (requires { descriptor.is_integer; }) r.integer = descriptor.is_integer;
            if constexpr (requires { descriptor.is_written; }) r.written = descriptor.is_written;
            if constexpr (requires { descriptor.has_secondary; }) {
                r.secondary = descriptor.has_secondary; r.secondary_bank = descriptor.secondary_cbuf_index;
                r.secondary_offset = descriptor.secondary_cbuf_offset; r.secondary_shift = descriptor.secondary_shift_left;
                r.shift = descriptor.shift_left;
            }
            if (descriptor.handle_program != ~0u) r.handle_program = program.info.handle_programs.at(descriptor.handle_program);
            ++binding; result.resources.push_back(std::move(r));
        }
    };
    append(program.info.texture_buffer_descriptors, ResourceKind::TextureBuffer);
    append(program.info.image_buffer_descriptors, ResourceKind::ImageBuffer);
    append(program.info.texture_descriptors, ResourceKind::Texture);
    append(program.info.image_descriptors, ResourceKind::Image);
    if (binding != bindings.unified) throw std::runtime_error("shader descriptor binding layout mismatch");
    return result;
}
uint32_t EvaluateHandle(const std::vector<uint32_t>& program, void* context,
    uint32_t (*read_cbuf)(void*, uint32_t, uint32_t),
    uint32_t (*read_storage)(void*, uint32_t, uint32_t)) {
    const auto handle = Shader::EvaluateHandleProgram(program,
        [&](uint32_t bank, uint32_t offset) { return read_cbuf(context, bank, offset); },
        [&](uint32_t bank, uint32_t offset) { return read_storage(context, bank, offset); });
    if (!handle) throw std::runtime_error("malformed shader texture handle expression");
    return *handle;
}
void DecodeAstc(const uint8_t* source, size_t source_size, uint32_t width, uint32_t height,
    uint32_t depth, uint32_t block_width, uint32_t block_height, uint8_t* output) {
    Tegra::Texture::ASTC::Decompress({source,source_size},width,height,depth,block_width,block_height,
        {output,static_cast<size_t>(width)*height*depth*4});
}
}
