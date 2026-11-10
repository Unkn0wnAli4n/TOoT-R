// The list of pipelines earlier runs created (<data>/pipeline_list.bin), kept so the next start can create them before
// the game runs (prewarm.cpp) instead of in the middle of play. A record holds everything a pipeline needs without the
// guest: references to translated shaders in the shader cache (key + identity of the translation) and the pipeline's
// fixed-function state, opaque here (render.cpp serializes it). Thread-safe.
#pragma once
#include <string>
#include <vector>

#include "../../rt.h"

namespace nvn::plist {

enum class Kind : u8 { Graphics = 0, Mesh = 1, Compute = 2 };
struct ShaderRef { u32 stage; u64 cache_key; u64 identity; };
struct Record {
    u64 id = 0;        // content hash of kind, shaders and state
    u32 uses = 0;      // runs that used the pipeline
    u64 last_used = 0; // unix seconds
    Kind kind = Kind::Graphics;
    std::vector<ShaderRef> shaders;
    std::vector<u8> state;
};

bool Enabled();  // false with OOT_NO_PREWARM=1
bool Trace();    // OOT_PREWARM_TRACE=1
u64 MakeId(Kind kind, const std::vector<ShaderRef>& shaders, const std::vector<u8>& state);
void Load(u32 shader_cache_version); // (once) reads the file; records of another format or shader cache version are dropped
size_t Count();
std::vector<Record> Snapshot();      // most valuable first (use count weighed by recency)
bool Add(Record record);             // a new pipeline: appended to the file; false when already listed
void Use(u64 id);                    // the pipeline was needed this run (once per run counts)
void Save(bool force);               // rewrites the file with updated counts, oldest/least used dropped (at most once a minute unless forced)

// Memory the machine can still give out right now (bytes); 0 when unknown.
u64 AvailableMemory();
u64 ProcessResidentBytes();

} // namespace nvn::plist
