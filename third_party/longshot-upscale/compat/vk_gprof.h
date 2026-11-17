#pragma once
#include <string>
#include "vk_internal.h"
namespace mkwvk { // (Longshot's per-pass GPU timing: not in this renderer)
inline bool gprof_begin(VkCommandBuffer) { return false; }
inline void gprof_end(VkCommandBuffer, const std::string&, int = 1, uint64_t = 0) {}
}
