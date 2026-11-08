// mkw-recomp: Eden's Common::unordered_set (boost's flat set there) as std::unordered_set: the API it uses
#pragma once

#include <functional>
#include <memory>
#include <unordered_set>

#include "common/container_hash.h"

namespace Common {
template <class Key, class Hash = std::hash<Key>, class Pred = std::equal_to<Key>, class Allocator = std::allocator<Key>>
using unordered_set = std::unordered_set<Key, Hash, Pred, Allocator>;
}
