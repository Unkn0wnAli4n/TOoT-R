// mkw-recomp: Eden's Common::unordered_map (boost's flat map there) as std::unordered_map: the API it uses
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>

#include "common/container_hash.h"

namespace Common {
template <class Key, class T, class Hash = std::hash<Key>, class Pred = std::equal_to<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
using unordered_map = std::unordered_map<Key, T, Hash, Pred, Allocator>;
}
