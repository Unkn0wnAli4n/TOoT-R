// This boundary keeps the shader library's types separate from the CPU runtime's IR types.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nvshader {
enum class ResourceKind { UniformBuffer, StorageBuffer, TextureBuffer, ImageBuffer, Texture, Image };
struct Resource {
    ResourceKind kind{};
    uint32_t binding{}, count{}, bank{}, offset{}, secondary_bank{}, secondary_offset{};
    uint32_t shift{}, secondary_shift{}, size_shift{}, texture_type{}, image_format{};
    uint32_t indirect_offset=~0u;
    uint32_t storage_index=~0u, storage_offset=0, storage_stride=0;
    bool sized_storage=false;
    bool secondary{}, depth{}, multisample{}, integer{}, written{};
    std::vector<uint32_t> handle_program;
};
// What a compiled variant assumed about state outside the shader code.
enum class DependencyKind : uint32_t {
    Value,   // the constant itself
    Mapped,  // whether the 64-bit address at bank:offset is mapped (value 0/1)
    Texture, // type and format of the texture whose handle is at bank:offset (value: type | format << 16)
    IndirectMapped,
};
struct Dependency { uint32_t bank, offset, value; DependencyKind kind = DependencyKind::Value; uint32_t indirect_offset=0; };
struct Request {
    uint32_t stage{}, binding_base{};
    const std::vector<uint8_t>* code{};
    const std::vector<uint8_t>* control{};
    void* context{};
    uint32_t (*read_constant)(void*, uint32_t, uint32_t){};
    bool (*mapped_address)(void*, uint64_t){};
    uint64_t (*read_pointer)(void*, uint64_t){};
    bool (*constant_pointer_available)(void*,uint32_t,uint32_t){};
    uint32_t (*texture_type)(void*, uint32_t){};
    uint32_t (*texture_format)(void*, uint32_t){};
    // Dependency handling for constants the compiler reads: kind of the word at bank:offset
    // (0 value, 1 texture handle, 2 not a dependency) and a description of a handle's texture.
    uint32_t (*classify_constant)(void*, uint32_t, uint32_t){};
    uint32_t (*describe_texture)(void*, uint32_t){};
    std::array<uint8_t, 32> input_types{}; // Float=0, Signed=1, Unsigned=2, Disabled=5
    std::array<uint8_t, 8> color_types{};
    std::array<uint64_t, 8> previous_stores{};
    bool have_previous_stores{}, convert_depth{}, negate_y{}, force_early_depth{}, dual_source{};
    uint32_t topology = 3; // Shader::InputTopology::Triangles
    float point_size = 0; // nonzero: the last pre-raster stage writes this point size
    // Task/mesh stages (6/7): SET_MESH_SHADER_A, SET_MESH_SHADER_B, SET_MESH_INIT_SHADER.
    std::array<uint32_t, 3> mesh_registers{};
    uint64_t storage_alignment = 16;
    uint64_t global_address_table = 0; // device address of the guest-address translation table
    bool buffer_float32_atomic_add=false;
    uint32_t vendor_id=0; // the GPU's (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel): translation workarounds for the others
    bool robust_buffer_access=true;
    // A stage the GPU runs with more lanes a subgroup than the 32 the game's programs are written for and cannot be held
    // to them (render.cpp PinStages): the most it may have. 0: 32 lanes.
    uint32_t unpinned_lanes=0;
    // Mesh shader fallback (a GPU whose task/mesh stages cannot run the game's programs): the task and mesh programs are
    // translated as compute programs that write vertex records into a buffer, and the mesh program's Result carries a
    // generated vertex stage that reads them back (render.cpp, "mesh fallback").
    bool mesh_fallback=false;
    bool mesh_fallback_task=false;        // mesh stage: a task program runs before it
    uint32_t mesh_fallback_task_threads=0; // that task program's threads a workgroup
    // Fragment stage after a mesh stage: generics the mesh program writes once a primitive, and whether it draws points
    // (a generic the fragment program reads and the mesh program never writes is then the point sprite's coordinate).
    uint32_t previous_per_primitive=0;
    bool previous_mesh=false, previous_points=false;
};
struct Result {
    std::vector<uint32_t> spirv;
    std::vector<Resource> resources;
    std::vector<Dependency> dependencies;
    std::array<uint64_t, 8> stores{};
    // Native global memory accesses left after lowering: they may touch any guest address.
    bool global_loads{}, global_stores{};
    // Task and mesh stages: threads a workgroup; mesh: its limits, the generics it writes per primitive, GL topology
    // (0 points, 1 lines, 4 triangles), and for the fallback the words of a vertex record and the generated vertex stage.
    uint32_t mesh_threads{}, mesh_payload_words{}, mesh_max_vertices{}, mesh_max_primitives{}, mesh_topology{}, mesh_per_primitive{}, mesh_record_words{};
    std::vector<uint32_t> fallback_vertex_spirv;
};
// The mesh fallback's buffers as the translated programs use them (the translator's emit_spirv.h; checked in the bridge).
// Records buffer: 8 header words - a VkDrawIndirectCommand, the record allocator, a VkDispatchIndirectCommand - then the records.
// Payload buffer: a map of MeshFallbackCapacity slot words, then each task workgroup's payload.
constexpr uint32_t MeshFallbackCapacity = 65535, MeshFallbackHeaderWords = 8, MeshFallbackRecordWords = 48u << 20, MeshFallbackDispatchWord = 5;
constexpr uint32_t MeshFirstPushOffset = 56, MeshFallbackPayloadPushOffset = 64, MeshFallbackRecordsPushOffset = 72, MeshFallbackIndirectPushOffset = 80;
Result Compile(const Request& request);
uint32_t EvaluateHandle(const std::vector<uint32_t>& program, void* context,
    uint32_t (*read_cbuf)(void*, uint32_t, uint32_t),
    uint32_t (*read_storage)(void*, uint32_t, uint32_t), uint32_t element = 0); // (element: of a handle table's descriptor array, Longshot's translator)
// The neural networks' matrix multiplies (HMMA 16x8x16 f16) on the GPU's cooperative matrices instead of in software.
// Which of a matrix's elements a lane holds is the driver's choice: `run` executes a 32-thread compute program (one
// workgroup) over one storage buffer of words (binding 0, in and out; false: it could not), twice - once to measure the
// device's element order, once to check a multiply against the CPU's. True: the translator uses them from now on
// (compute programs translated after this differ: render.cpp mixes NativeMatrixMultiply() into their cache keys).
bool ProbeCooperativeMatrices(bool (*run)(const std::vector<uint32_t>& spirv, std::vector<uint32_t>& words), std::string* report);
bool NativeMatrixMultiply();
void DecodeAstc(const uint8_t* source, size_t source_size, uint32_t width, uint32_t height,
    uint32_t depth, uint32_t block_width, uint32_t block_height, uint8_t* output);
}
