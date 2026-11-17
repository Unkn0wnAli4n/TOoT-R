#pragma once
#include <stdint.h>
/* (from Longshot gpu/vk/vk_api.h) One completed temporal-upscale sample; all zero while inactive. */
typedef struct MkwVkJitter { uint32_t width, height, phases; } MkwVkJitter;
