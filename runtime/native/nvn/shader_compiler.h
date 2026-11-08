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
};
struct Result {
    std::vector<uint32_t> spirv;
    std::vector<Resource> resources;
    std::vector<Dependency> dependencies;
    std::array<uint64_t, 8> stores{};
    // Native global memory accesses left after lowering: they may touch any guest address.
    bool global_loads{}, global_stores{};
};
Result Compile(const Request& request);
uint32_t EvaluateHandle(const std::vector<uint32_t>& program, void* context,
    uint32_t (*read_cbuf)(void*, uint32_t, uint32_t),
    uint32_t (*read_storage)(void*, uint32_t, uint32_t));
void DecodeAstc(const uint8_t* source, size_t source_size, uint32_t width, uint32_t height,
    uint32_t depth, uint32_t block_width, uint32_t block_height, uint8_t* output);
}
