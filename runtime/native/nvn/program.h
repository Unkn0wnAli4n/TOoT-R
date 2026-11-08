#pragma once
#include <array>
#include "vk.h"

namespace nvn {
struct ShaderCode {
    u64 address = 0;
    u32 stage = 0; // NVN: vertex, fragment, geometry, tess-control, tess-eval, compute
    std::vector<u8> control;
    std::vector<u8> code;
    std::vector<u8> constants; // compiler literals, bound by NVN in hardware constant bank 1
};
struct Program {
    std::vector<ShaderCode> shaders;
    bool sample_shading = false;
};
Program* HostProgram(u64 object);
}
