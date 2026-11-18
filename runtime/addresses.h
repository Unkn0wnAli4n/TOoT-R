// Addresses in the game's own code and data.
//
// A hook, a cheat or a tool that reads the game's memory names a place in ONE game version: update v131072, the
// reference this project was made with. Another update moves them. tools/signatures.py finds each place in the game
// being built by what its code looks like, and the build writes <data>/addresses.txt; here a place is asked for by its
// offset in the reference version and comes back where it is in this game - or 0 when this version has no such place,
// and whatever used it stays off.
#pragma once
#include <cstdint>

namespace rt {
enum class GameModule { main, subsdk0, sdk };
// The absolute address of the place at `reference_offset` of the reference version's module; 0: not in this game.
uint64_t AddressOf(uint64_t reference_offset, GameModule module = GameModule::main);
// How many places were asked for and not found (the settings window says which features are off).
unsigned AddressesMissing();
} // namespace rt

// OOT_AT(0x4c7e890): the place at main + 0x4c7e890 of the reference version, in this game (looked up once).
#define OOT_AT(offset) ([]() -> uint64_t { static const uint64_t address = rt::AddressOf(offset); return address; }())
#define OOT_AT_IN(module, offset) ([]() -> uint64_t { static const uint64_t address = rt::AddressOf(offset, rt::GameModule::module); return address; }())
