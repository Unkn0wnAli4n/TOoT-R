// Shader programs keep the GPU code and control metadata; pipelines compile them
// with the textures, constant buffers and render state bound at each draw.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <atomic>
#include "../../native.h"
#include "program.h"
#include "memory.h"
using namespace rt;
using namespace nvn;
namespace nvn {
Program* HostProgram(u64 object) { return *reinterpret_cast<Program**>(object); }
}
namespace {
struct ShaderData { u64 address, control; };
bool SetShaders(Cpu* c, Program& program, s32 count, const ShaderData* data) {
    if (count < 0 || count > 6 || (count && !data)) return false;
    std::vector<ShaderCode> shaders;
    const u64 scratch = ArenaAlloc(9 * 0x28);
    for (s32 i = 0; i < count; ++i) {
        if (!data[i].control || !LookupAddress(data[i].address).pool) return false;
        const u32* control = reinterpret_cast<const u32*>(data[i].control);
        if (control[0] != 0x98761234) Fatal("unknown NVN shader control magic %#x", control[0]);
        // This SDK helper constructs the appropriate control-reader for the exact
        // control version. It only reads metadata and does not create GPU objects.
        const u64 object = CallGuest(c, SdkBase + 0x663ab0, {scratch, data[i].control});
        if (!object) return false;
        const u64 vtable = *reinterpret_cast<const u64*>(object);
        const u32 stage = CallGuest(c, *reinterpret_cast<const u64*>(vtable + 0xa8), {object});
        if (stage > 8) Fatal("unknown NVN shader stage %u", stage);
        ShaderCode shader;
        shader.address = data[i].address; shader.stage = stage;
        const u32 control_size = CallGuest(c, *reinterpret_cast<const u64*>(vtable), {object});
        const u32 code_size = CallGuest(c, *reinterpret_cast<const u64*>(vtable + 0x60), {object});
        if (control_size < 0x80 || control_size > 0x10000 || !code_size || code_size > 0x100000)
            Fatal("invalid NVN shader control/code extent %#x/%#x", control_size, code_size);
        // The reader reports only the fixed header; newer control versions keep reflection
        // sections after it, with the whole size at +0x30.
        u32 full_size = control_size;
        if (control[0x30 / 4] > control_size && control[0x30 / 4] <= 0x10000) full_size = control[0x30 / 4];
        shader.control.assign(reinterpret_cast<const u8*>(control), reinterpret_cast<const u8*>(control) + full_size);
        const auto backing = LookupAddress(shader.address);
        if (code_size > backing.available) Fatal("NVN shader code crosses unmapped pool storage");
        shader.code.assign(reinterpret_cast<const u8*>(backing.cpu_address), reinterpret_cast<const u8*>(backing.cpu_address) + code_size);
        const u32 constant_size=CallGuest(c,*reinterpret_cast<const u64*>(vtable+0x68),{object});
        if(constant_size){
            const u32 constant_offset=CallGuest(c,*reinterpret_cast<const u64*>(vtable+0x70),{object});
            if(constant_size>0x10000||constant_offset>0x100000)Fatal("invalid shader literal extent");
            // Literals follow aligned GPU instruction storage; these offsets are
            // distinct from reflection/debug sections inside the CPU control blob.
            const auto literal_backing=LookupAddress(shader.address+constant_offset);
            if(!literal_backing.pool||constant_size>literal_backing.available)Fatal("shader literals cross unmapped storage");
            const auto* literals=reinterpret_cast<const u8*>(literal_backing.cpu_address);
            shader.constants.assign(literals,literals+constant_size);
        }
        if (const char* dir = std::getenv("OOT_SHADER_CONTROL_DUMP")) {
            char path[512]; std::snprintf(path, sizeof path, "%s/%llx-stage%u.control", dir, (unsigned long long)shader.address, stage);
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(shader.control.data(), 1, shader.control.size(), f); std::fclose(f); }
            std::snprintf(path, sizeof path, "%s/%llx-stage%u.code", dir, (unsigned long long)shader.address, stage);
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(shader.code.data(), 1, shader.code.size(), f); std::fclose(f); }
            const u32 blob = std::max<u32>(code_size, control[0x44 / 4]);
            const auto whole = LookupAddress(shader.address);
            std::snprintf(path, sizeof path, "%s/%llx-stage%u.blob", dir, (unsigned long long)shader.address, stage);
            if (blob <= whole.available) if (FILE* f = std::fopen(path, "wb")) { std::fwrite(reinterpret_cast<const void*>(whole.cpu_address), 1, blob, f); std::fclose(f); }
        }
        if (std::getenv("OOT_SHADER_TRACE")) {
            RT_LOG("shader stage %u address %#llx control version %u.%u vtable %#llx", stage,
                   (unsigned long long)shader.address, control[1], control[2], (unsigned long long)vtable);
            for (u32 slot = 0; slot <= 0xc0; slot += 8)
                RT_LOG("  control vtable +%#x = %#llx", slot, (unsigned long long)*reinterpret_cast<u64*>(vtable + slot));
            for (u32 word = 0; word < 0x100 / 4; word += 4)
                RT_LOG("  control +%#x: %08x %08x %08x %08x", word * 4, control[word], control[word+1], control[word+2], control[word+3]);
            const auto range = LookupAddress(shader.address);
            const auto* code = reinterpret_cast<const u32*>(range.cpu_address);
            for (u32 word = 0; word < 0x100 / 4 && word * 4 + 16 <= range.available; word += 4)
                RT_LOG("  code +%#x: %08x %08x %08x %08x", word * 4, code[word], code[word+1], code[word+2], code[word+3]);
        }
        shaders.push_back(std::move(shader));
    }
    program.shaders = std::move(shaders);
    return true;
}
}
NATIVE("nvnProgramInitialize") {
    *Ptr<Program*>(c, 0) = new Program{}; Ret(c, 1);
}
NATIVE("nvnProgramInitializeFromBuilder") {
    const u64 object = c->x[0];
    const u8* builder = Ptr<u8>(c, 1);
    u32 count, sample; u64 shaders;
    std::memcpy(&sample, builder + 0x18, 4); std::memcpy(&count, builder + 0x28, 4);
    std::memcpy(&shaders, builder + 0x30, 8);
    auto* program = new Program{}; program->sample_shading = sample != 0;
    const bool ok = SetShaders(c, *program, count, reinterpret_cast<const ShaderData*>(shaders));
    if (!ok) { delete program; Ret(c, 0); return; }
    *reinterpret_cast<Program**>(object) = program; Ret(c, 1);
}
NATIVE("nvnProgramSetShaders") {
    Program* program = HostProgram(c->x[0]); const s32 count = c->x[1]; const auto* data = Ptr<ShaderData>(c, 2);
    Ret(c, SetShaders(c, *program, count, data));
}
NATIVE("nvnProgramSetShadersExt") {
    const u64 object=c->x[0];const s32 count=c->x[1];const u8* input=Ptr<u8>(c,2);
    if(count<0||count>6){Ret(c,0);return;}
    std::vector<ShaderData> data(count);
    for(s32 i=0;i<count;++i)std::memcpy(&data[i],input+i*24,sizeof(ShaderData));
    Ret(c,SetShaders(c,*HostProgram(object),count,data.data()));
}
NATIVE("nvnProgramFinalize") { delete HostProgram(c->x[0]); *Ptr<Program*>(c, 0) = nullptr; }
NATIVE("nvnProgramSetDebugLabel") {}
NATIVE("nvnProgramSetSampleShading") { HostProgram(c->x[0])->sample_shading = c->x[1] != 0; }
// Host Vulkan pipelines allocate their own scratch/compute memory.
NATIVE("nvnProgramGetScratchBytesPerWarp") { Ret(c, 0); }
NATIVE("nvnProgramGetDeviceComputeMemorySize") { Ret(c, 0); }
