#pragma once
#include <cstddef>
#include <cstdint>
namespace mkwvk { struct BuiltinSpv { const uint32_t* code; size_t size; };
inline BuiltinSpv builtin_spv(const char*, const uint32_t* words, size_t size) { return {words, size}; } }
#define BUILTIN_SPV(words) ::mkwvk::builtin_spv(#words, words, sizeof words)
#define BUILTIN_SPV_ARGS(words) BUILTIN_SPV(words).code, BUILTIN_SPV(words).size
