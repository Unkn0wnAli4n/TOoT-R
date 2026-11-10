// Pipeline prewarming: before the game starts, the pipelines earlier runs created (pipeline_list.h) are made again on
// background threads, with a progress screen in the game window, so play does not stop for them the first time an area is
// entered. OOT_NO_PREWARM=1 disables it, OOT_PREWARM_TRACE=1 logs counts and timings.
#pragma once
#include <string>

#include "pipeline_list.h"

namespace nvn {

// main(): after the modules are loaded, before the guest runs. Brings up Vulkan and the window only when there is something to make
// (the first run, with no list, shows nothing and returns at once).
void PrewarmMain();

// render.cpp
bool PrewarmBuild(const plist::Record& record);  // creates one pipeline and enters it in the tables draws look in
u64 ShaderCacheBytes(u64 key);                   // size of the shader cache file of a key (what its translations take in memory)
std::string ShaderCacheDir();
u32 ShaderCacheVersionValue();

} // namespace nvn
