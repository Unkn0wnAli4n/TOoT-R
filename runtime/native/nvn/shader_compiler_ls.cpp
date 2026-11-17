// Native Ampere instructions -> SPIR-V through Longshot's translator (third_party/shader-ls; built with -DOOT_SHADER=longshot).
// The same interface as shader_compiler.cpp; what differs in the translators' contracts is mapped here.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "shader_compiler.h"
#include <sirit/sirit.h>
#include "oot_ir_passes.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/mma_layout.h"
#include "shader_recompiler/backend/nvn_storage_descriptor.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
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
bool native_mma = false; // the device's cooperative matrices run the console's multiplies (ProbeCooperativeMatrices)
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
    bool IsCbufIndirectStorageBufferPointer(uint32_t bank,uint32_t offset,uint32_t indirect) {
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
        (void)complete; // (as Longshot: a program whose end the capture does not see is lowered as far as its memory goes)
        std::vector<uint32_t> traps;
        for (uint32_t i = 0; i < native.size(); ++i) if (native[i].Bits(0, 12) == 0x95c) traps.push_back(i * 16);
        // The neural upscaler's matrix instructions: kept as HMMA_nx2 and run in software by the back end (mma_layout.h).
        // Per GPU vendor, as Longshot sets them (vk_coopmat.cpp); OOT_NAN_GUARD=0, OOT_POSITION_INVARIANT=0|1,
        // OOT_FETCH_BOUNDS=0|1 override.
        {
            const bool nvidia = request.vendor_id == 0x10DE;
            const auto flag = [](const char* name, bool otherwise) { const char* e = std::getenv(name); return e ? std::atoi(e) != 0 : otherwise; };
            // float colour outputs carry no NaN or infinity to their target: one NaN texel spreads through bloom into glowing blocks
            Shader::Backend::SPIRV::frag_nan_guard = flag("OOT_NAN_GUARD", true);
            // the depth pre-pass and the EQUAL colour pass must compute one position: only NVIDIA's compiler does without being told
            Shader::Backend::SPIRV::position_invariant = flag("OOT_POSITION_INVARIANT", !nvidia);
            // an out-of-bounds texel fetch answers zero, as the console's GPU does (pale boxes at the screen's edges on AMD)
            Shader::Backend::SPIRV::fetch_bounds = flag("OOT_FETCH_BOUNDS", !nvidia);
            Shader::Backend::SPIRV::fetch_bounds_buffers = Shader::Backend::SPIRV::fetch_bounds && !request.robust_buffer_access;
            // storage buffer tables searched in a loop: Mesa inlines the written-out search into 100k instructions (Intel)
            Shader::Backend::SPIRV::global_table_loop = flag("OOT_TABLE_LOOP", request.vendor_id == 0x8086);
            // isnan as a test of the bits (AMD's older Windows shader compiler folds OpIsNan away): opt-in, OOT_NAN_BITS=1
            Shader::Backend::SPIRV::nan_bits = flag("OOT_NAN_BITS", false);
        }
        Shader::Ampere::native_matrix_multiply = stage == Stage::Compute;
        Shader::Backend::SPIRV::mma_software = !native_mma; // (ProbeCooperativeMatrices)
        const auto lowered = Shader::Ampere::LowerToMaxwell(native);
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

namespace sp = Shader::Backend::SPIRV;
static_assert(MeshFallbackCapacity == sp::MESH_FALLBACK_CAPACITY && MeshFallbackHeaderWords == sp::MESH_FALLBACK_HEADER_WORDS && MeshFallbackRecordWords == sp::MESH_FALLBACK_RECORD_WORDS &&
              MeshFallbackDispatchWord == sp::MESH_FALLBACK_DISPATCH_WORD && sp::MESH_FALLBACK_DRAW_WORD == 0 && MeshFirstPushOffset == sp::MESH_FIRST_PUSH_OFFSET &&
              MeshFallbackPayloadPushOffset == sp::MESH_FALLBACK_PAYLOAD_PUSH_OFFSET && MeshFallbackRecordsPushOffset == sp::MESH_FALLBACK_RECORDS_PUSH_OFFSET &&
              MeshFallbackIndirectPushOffset == sp::MESH_FALLBACK_INDIRECT_INPUT_PUSH_OFFSET && sp::GUEST_PAGES_PUSH_OFFSET == 48, "render.cpp's copy of the mesh fallback layout");
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
    host.min_ssbo_alignment = request.storage_alignment; // (bindings start at the device's alignment below the pointer)
    host.support_handle_table_arrays = true; // (a handle read from a table at a per-invocation index: a descriptor array)
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
    const bool fallback = request.mesh_fallback && (program.stage == Stage::Task || program.stage == Stage::Mesh);
    if (fallback) {
        runtime.mesh_fallback = true; runtime.mesh_fallback_capacity = Shader::Backend::SPIRV::MESH_FALLBACK_CAPACITY;
        if (program.stage == Stage::Mesh) { runtime.mesh_fallback_task = request.mesh_fallback_task; runtime.mesh_fallback_task_threads = request.mesh_fallback_task_threads; }
    }
    if (program.stage == Stage::Fragment && request.previous_mesh) {
        // (what the mesh program writes once a primitive: PerPrimitiveEXT inputs; behind the fallback's vertex stage the same values flat)
        for (uint32_t i = 0; i < 32; ++i) if (request.previous_per_primitive >> i & 1) {
            if (request.mesh_fallback) program.info.interpolation[i] = Shader::Interpolation::Flat; else runtime.per_primitive_generics.set(i);
        }
        // (points: a generic the fragment program reads and the mesh program never writes is the point sprite's coordinate,
        // as the hardware fills it in: the grass shadow reads its alpha mask there. Longshot's finding.)
        if (request.previous_points && request.have_previous_stores)
            for (size_t i = 0; i < Shader::IR::NUM_GENERICS; ++i)
                if (program.info.loads.Generic(i) && !runtime.previous_stage_stores.Generic(i)) runtime.point_coord_generics.set(i);
    }
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
    profile.support_int64 = profile.support_int16 = profile.support_int8 = true;
    profile.support_vote = profile.support_shuffle_relative = true;
    profile.support_sampled_image_array_nonuniform_indexing = true;
    profile.support_viewport_index_layer_non_geometry = profile.support_multi_viewport = true;
    profile.max_subgroup_size = 32; profile.max_user_clip_distances = 8;
    // (a stage running wider than the guest's 32 lanes: the translator's long path for the lane operations, Longshot's
    // profile_graphics)
    if (request.unpinned_lanes > 32) { profile.warp_size_potentially_larger_than_guest = true; profile.max_subgroup_size = request.unpinned_lanes; }
    // (gathers nudged by 1/512 texel so the four texels picked at a texel's edge are the console's: shadow edges came out
    // as dashed lines on AMD and Intel without it, Longshot's vk_vendor.h. OOT_GATHER_OFFSET=0|1)
    { const char* e = std::getenv("OOT_GATHER_OFFSET"); profile.need_gather_subpixel_offset = e && std::atoi(e) != 0; } // (opt-in until it has run on an AMD GPU here)
    // Subgroup operations in every stage including task and mesh (Shader::Stage 7/8; the profile's
    // default 0x7F left them out): without them ballots/votes/shuffles fall back to single-lane
    // results, so mesh programs computed their primitive count per lane (no grass drawn) and task
    // programs launched no mesh workgroups. The host GPU supports subgroups there (checked in vk.cpp).
    profile.supported_subgroup_stages = 0x1FF;
    profile.support_typeless_image_loads = true;
    profile.supported_spirv = 0x00010600; profile.support_descriptor_aliasing = true;
    profile.min_ssbo_alignment = request.storage_alignment;
    profile.support_uniform_and_storage_buffer_8bit = profile.support_storage_buffer_8bit = true;
    profile.support_uniform_and_storage_buffer_16bit = profile.support_storage_buffer_16bit = true;
    profile.support_float_controls = true; profile.support_int64_atomics = true; profile.support_derivative_control = true;
    profile.support_separate_denorm_behavior = profile.support_separate_rounding_mode = true;
    profile.support_fp32_denorm_preserve = profile.support_fp32_denorm_flush = true;
    profile.support_fp16_signed_zero_nan_preserve = profile.support_fp32_signed_zero_nan_preserve = true;
    profile.support_demote_to_helper_invocation = true;
    profile.support_float32_atomic_add = request.buffer_float32_atomic_add;
    Shader::Backend::Bindings bindings{}; bindings.unified = request.binding_base;
    Result result; result.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
    result.dependencies = std::move(env.dependencies);
    if (program.stage == Stage::Task || program.stage == Stage::Mesh) result.mesh_threads = program.workgroup_size[0];
    if (program.stage == Stage::Mesh) {
        const Shader::MeshInfo& mesh = program.mesh;
        result.mesh_payload_words = mesh.payload_words; result.mesh_max_vertices = mesh.max_vertices; result.mesh_max_primitives = mesh.max_primitives; result.mesh_topology = mesh.topology;
        result.mesh_record_words = 1 + (uint32_t)mesh.vertex_lines.size() + (uint32_t)mesh.primitive_lines.size();
        for (const uint32_t line : mesh.primitive_lines) { const auto a = static_cast<Shader::IR::Attribute>(line); if (Shader::IR::IsGeneric(a)) result.mesh_per_primitive |= 1u << Shader::IR::GenericAttributeIndex(a); }
        if (fallback) {
            // The generated vertex stage: no descriptors of its own (its input is the records), the mesh program's outputs.
            Shader::RuntimeInfo vertex{}; vertex.mesh_fallback = vertex.mesh_fallback_vertex = true;
            vertex.mesh_fallback_capacity = Shader::Backend::SPIRV::MESH_FALLBACK_CAPACITY; vertex.convert_depth_mode = request.convert_depth;
            for (uint32_t i = 0; i < 32; ++i) if (result.mesh_per_primitive >> i & 1) vertex.per_primitive_generics.set(i);
            Shader::Info kept{std::move(program.info)}; program.info = Shader::Info{}; program.info.stores = kept.stores;
            program.stage = Stage::VertexB;
            Shader::Backend::Bindings none{}; none.unified = bindings.unified;
            result.fallback_vertex_spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, vertex, program, none);
            program.info = std::move(kept); program.stage = Stage::Mesh;
        }
    }
    if (program.stage == Stage::Task) result.mesh_payload_words = program.mesh.payload_words;
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
        // (a storage buffer table, count > 1: one binding holding an array of that many buffers)
        r.written = descriptor.is_written; binding += r.count > 1 ? 1 : r.count; result.resources.push_back(std::move(r));
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
    uint32_t (*read_storage)(void*, uint32_t, uint32_t), uint32_t element) {
    const auto handle = Shader::EvaluateHandleProgram(program,
        [&](uint32_t bank, uint32_t offset) { return read_cbuf(context, bank, offset); },
        [&](uint32_t bank, uint32_t offset) { return read_storage(context, bank, offset); }, element);
    if (!handle) throw std::runtime_error("malformed shader texture handle expression");
    return *handle;
}
// ---- cooperative matrices -------------------------------------------------------------------------------------------
// As Longshot's vk_coopmat.cpp does it: a lane's fragment registers hold its share of each matrix in the console's order
// (PTX mma.m16n8k16: two f16 a register, low half first); which positions a lane's elements are on this device is
// measured with matrices whose values are their positions, and a multiply in that order is checked against the CPU's.
namespace {
uint16_t ToHalf(float f) { // (small integers only: exact)
    if (f == 0.0f) return 0;
    uint32_t x; std::memcpy(&x, &f, 4);
    return (uint16_t)(((x >> 16) & 0x8000) | (uint32_t)(((int)((x >> 23) & 0xff) - 127 + 15) << 10) | ((x >> 13) & 0x3ff));
}
float FromHalf(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    if (!exp) return 0.0f; // (the probe's values are zero or normal)
    const uint32_t x = sign | ((exp - 15 + 127) << 23) | (man << 13); float f; std::memcpy(&f, &x, 4); return f;
}
// The console's fragment positions: lane t (group t / 4, index t % 4), element e -> (row, column).
void ConsoleA(int t, int e, int& r, int& k) { r = t / 4 + ((e >> 1) & 1) * 8; k = 2 * (t % 4) + (e & 1) + (e >> 2) * 8; }
void ConsoleB(int t, int e, int& k, int& n) { k = 2 * (t % 4) + (e & 1) + (e >> 1) * 8; n = t / 4; }
void ConsoleC(int t, int e, int& r, int& n) { r = t / 4 + (e >> 1) * 8; n = 2 * (t % 4) + (e & 1); }
struct MatrixProgram {
    Sirit::Module m{0x00010600};
    Sirit::Id u32, f16, f16x2, buffer, word_pointer, lane, a, b, c, zero;
    Sirit::Id C(uint32_t v) { return m.Constant(u32, v); }
    Sirit::Id WordPointer(Sirit::Id index) { return m.OpAccessChain(word_pointer, buffer, C(0), index); }
    Sirit::Id Word(Sirit::Id index) { return m.OpLoad(u32, WordPointer(index)); }
    void Store(Sirit::Id index, Sirit::Id value) { m.OpStore(WordPointer(index), value); }
    Sirit::Id Plus(Sirit::Id base, uint32_t add) { return m.OpIAdd(u32, base, C(add)); }
    Sirit::Id Pack(Sirit::Id lo, Sirit::Id hi) { return m.OpBitcast(u32, m.OpCompositeConstruct(f16x2, lo, hi)); }
    Sirit::Id Element(Sirit::Id matrix, uint32_t j) { return m.OpCompositeExtract(f16, matrix, j); }
};
template <typename Body> std::vector<uint32_t> BuildMatrixProgram(Body&& body) {
    MatrixProgram s; auto& m = s.m;
    m.AddCapability(spv::Capability::Shader); m.AddCapability(spv::Capability::Float16);
    m.AddCapability(spv::Capability::CooperativeMatrixKHR); m.AddExtension("SPV_KHR_cooperative_matrix");
    m.AddCapability(spv::Capability::VulkanMemoryModel); m.AddExtension("SPV_KHR_vulkan_memory_model");
    m.SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::VulkanKHR);
    const Sirit::Id void_type = m.TypeVoid(); s.u32 = m.TypeUInt(32);
    const Sirit::Id uvec3 = m.TypeVector(s.u32, 3); s.f16 = m.TypeFloat(16); s.f16x2 = m.TypeVector(s.f16, 2);
    const Sirit::Id words = m.TypeRuntimeArray(s.u32); m.Decorate(words, spv::Decoration::ArrayStride, 4U);
    const Sirit::Id block = m.TypeStruct(words); m.Decorate(block, spv::Decoration::Block); m.MemberDecorate(block, 0U, spv::Decoration::Offset, 0U);
    s.buffer = m.AddGlobalVariable(m.TypePointer(spv::StorageClass::StorageBuffer, block), spv::StorageClass::StorageBuffer);
    m.Decorate(s.buffer, spv::Decoration::DescriptorSet, 0U); m.Decorate(s.buffer, spv::Decoration::Binding, 0U);
    const Sirit::Id invocation = m.AddGlobalVariable(m.TypePointer(spv::StorageClass::Input, uvec3), spv::StorageClass::Input);
    m.Decorate(invocation, spv::Decoration::BuiltIn, spv::BuiltIn::LocalInvocationId);
    s.word_pointer = m.TypePointer(spv::StorageClass::StorageBuffer, s.u32);
    const Sirit::Id scope = s.C((uint32_t)spv::Scope::Subgroup);
    const auto matrix = [&](uint32_t rows, uint32_t columns, spv::CooperativeMatrixUse use) { return m.TypeCooperativeMatrixKHR(s.f16, scope, s.C(rows), s.C(columns), s.C((uint32_t)use)); };
    s.a = matrix(16, 16, spv::CooperativeMatrixUse::MatrixAKHR); s.b = matrix(16, 8, spv::CooperativeMatrixUse::MatrixBKHR);
    s.c = matrix(16, 8, spv::CooperativeMatrixUse::MatrixAccumulatorKHR); s.zero = m.Constant(s.f16, 0U);
    const Sirit::Id main = m.OpFunction(void_type, spv::FunctionControlMask::MaskNone, m.TypeFunction(void_type));
    m.AddLabel();
    s.lane = m.OpCompositeExtract(s.u32, m.OpLoad(uvec3, invocation), 0U);
    body(s);
    m.OpReturn(); m.OpFunctionEnd();
    m.AddEntryPoint(spv::ExecutionModel::GLCompute, main, "main", s.buffer, invocation);
    m.AddExecutionMode(main, spv::ExecutionMode::LocalSize, 32U, 1U, 1U);
    return m.Assemble();
}
}
bool NativeMatrixMultiply() { return native_mma; }
bool ProbeCooperativeMatrices(bool (*run)(const std::vector<uint32_t>&, std::vector<uint32_t>&), std::string* report) {
    namespace sp = Shader::Backend::SPIRV;
    const auto say = [&](const std::string& text) { if (report) *report = text; return false; };
    try {
    // 1. The order: A (16x16), B (16x8), C (16x8) loaded row major from words 0, 128, 192, each element's value its
    // position; lane t writes its 8 A elements at 256 + 4t, its 4 B at 384 + 2t, its 4 C at 448 + 2t.
    std::vector<uint32_t> data(512, 0);
    const auto put = [&](uint32_t base, int rows, int columns) { for (int r = 0; r < rows; ++r) for (int c = 0; c < columns; c += 2)
        data[base + (r * columns + c) / 2] = (uint32_t)ToHalf((float)(r * columns + c)) | ((uint32_t)ToHalf((float)(r * columns + c + 1)) << 16); };
    put(0, 16, 16); put(128, 16, 8); put(192, 16, 8);
    const auto measure = BuildMatrixProgram([](MatrixProgram& s) { auto& m = s.m;
        const auto load = [&](Sirit::Id type, uint32_t base, uint32_t stride) { return m.OpCooperativeMatrixLoadKHR(type, s.WordPointer(s.C(base)), s.C(0) /* row major */, s.C(stride)); };
        const Sirit::Id a = load(s.a, 0, 8), b = load(s.b, 128, 4), c = load(s.c, 192, 4);
        const auto out = [&](Sirit::Id matrix, uint32_t base, uint32_t words) { const Sirit::Id at = m.OpIAdd(s.u32, s.C(base), m.OpIMul(s.u32, s.lane, s.C(words)));
            for (uint32_t w = 0; w < words; ++w) s.Store(s.Plus(at, w), s.Pack(s.Element(matrix, 2 * w), s.Element(matrix, 2 * w + 1))); };
        out(a, 256, 4); out(b, 384, 2); out(c, 448, 2); });
    if (!run(measure, data)) return say("the measuring program did not run");
    const auto device = [&](uint32_t base, uint32_t words, int t, int j) { const uint32_t w = data[base + t * words + j / 2]; return (int)FromHalf((uint16_t)(j & 1 ? w >> 16 : w & 0xffff)); };
    const auto find = [&](uint32_t base, int count, int columns, auto console, auto& order) {
        for (int e = 0; e < count; ++e) { int found = -1;
            for (int t = 0; t < 32; ++t) { int r, c; console(t, e, r, c); int j = 0;
                while (j < count && device(base, count / 2, t, j) != r * columns + c) ++j;
                if (j == count || (found >= 0 && j != found)) return false;
                found = j; }
            order[e] = (uint32_t)found; }
        return true; };
    if (!find(256, 8, 16, ConsoleA, sp::mma_order_a) || !find(384, 4, 8, ConsoleB, sp::mma_order_b) || !find(448, 4, 8, ConsoleC, sp::mma_order_c))
        return say("a lane's elements are not the console's positions");
    // 2. A multiply as the translator emits it (emit_spirv_warp.cpp): lane t's fragment words at 8t (A 4, B 2, C 2), D's 2 at 256 + 2t.
    const auto av = [](int r, int k) { return (float)(((r * 7 + k * 3 + 1) % 5) - 2); };
    const auto bv = [](int k, int n) { return (float)(((k * 5 + n * 11 + 2) % 4) - 1); };
    const auto cv = [](int r, int n) { return (float)(((r * 3 + n * 13) % 9) - 4); };
    data.assign(320, 0xffffffffu);
    const auto pack = [](float lo, float hi) { return (uint32_t)ToHalf(lo) | ((uint32_t)ToHalf(hi) << 16); };
    for (int t = 0; t < 32; ++t) { int r0, c0, r1, c1;
        for (int e = 0; e < 8; e += 2) { ConsoleA(t, e, r0, c0); ConsoleA(t, e + 1, r1, c1); data[t * 8 + e / 2] = pack(av(r0, c0), av(r1, c1)); }
        for (int e = 0; e < 4; e += 2) { ConsoleB(t, e, r0, c0); ConsoleB(t, e + 1, r1, c1); data[t * 8 + 4 + e / 2] = pack(bv(r0, c0), bv(r1, c1)); }
        for (int e = 0; e < 4; e += 2) { ConsoleC(t, e, r0, c0); ConsoleC(t, e + 1, r1, c1); data[t * 8 + 6 + e / 2] = pack(cv(r0, c0), cv(r1, c1)); } }
    const auto multiply = BuildMatrixProgram([](MatrixProgram& s) { auto& m = s.m; namespace sp = Shader::Backend::SPIRV;
        const Sirit::Id base = m.OpIMul(s.u32, s.lane, s.C(8));
        const auto load = [&](Sirit::Id type, uint32_t first, uint32_t count, const auto& order) { Sirit::Id matrix = m.OpCompositeConstruct(type, s.zero);
            for (uint32_t w = 0; w < count; ++w) { const Sirit::Id pair = m.OpBitcast(s.f16x2, s.Word(s.Plus(base, first + w)));
                for (uint32_t h = 0; h < 2; ++h) matrix = m.OpCompositeInsert(type, m.OpCompositeExtract(s.f16, pair, h), matrix, order[w * 2 + h]); }
            return matrix; };
        const Sirit::Id d = m.OpCooperativeMatrixMulAddKHR(s.c, load(s.a, 0, 4, sp::mma_order_a), load(s.b, 4, 2, sp::mma_order_b), load(s.c, 6, 2, sp::mma_order_c));
        const Sirit::Id out = m.OpIAdd(s.u32, s.C(256), m.OpIMul(s.u32, s.lane, s.C(2)));
        for (uint32_t w = 0; w < 2; ++w) s.Store(s.Plus(out, w), s.Pack(s.Element(d, sp::mma_order_c[w * 2]), s.Element(d, sp::mma_order_c[w * 2 + 1]))); });
    if (!run(multiply, data)) return say("the test multiply did not run");
    int bad = 0;
    for (int t = 0; t < 32; ++t) for (int e = 0; e < 4; ++e) { int r, n; ConsoleC(t, e, r, n);
        float want = cv(r, n); for (int k = 0; k < 16; ++k) want += av(r, k) * bv(k, n);
        const uint32_t w = data[256 + t * 2 + e / 2];
        if (FromHalf((uint16_t)(e & 1 ? w >> 16 : w & 0xffff)) != want) ++bad; }
    if (bad) return say(std::to_string(bad) + " of 128 elements of the test multiply differ from the CPU's");
    } catch (const std::exception& error) { return say(std::string("probe failed: ") + error.what()); }
    native_mma = true;
    if (report) { *report = "element orders A"; for (auto v : sp::mma_order_a) *report += " " + std::to_string(v);
        *report += ", B"; for (auto v : sp::mma_order_b) *report += " " + std::to_string(v); *report += ", C"; for (auto v : sp::mma_order_c) *report += " " + std::to_string(v); }
    return true;
}
void DecodeAstc(const uint8_t* source, size_t source_size, uint32_t width, uint32_t height,
    uint32_t depth, uint32_t block_width, uint32_t block_height, uint8_t* output) {
    Tegra::Texture::ASTC::Decompress({source,source_size},width,height,depth,block_width,block_height,
        {output,static_cast<size_t>(width)*height*depth*4});
}
}
