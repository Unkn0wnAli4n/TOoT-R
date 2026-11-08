// Newer fmt no longer formats enums implicitly; the shader compiler's diagnostics print raw enum
// values. Enums without their own formatter print as their underlying integer.
#pragma once
#include <type_traits>
#include <fmt/format.h>
template <typename E>
    requires std::is_enum_v<E>
struct fmt::formatter<E, char> : fmt::formatter<std::underlying_type_t<E>, char> {
    auto format(E value, fmt::format_context& ctx) const {
        return fmt::formatter<std::underlying_type_t<E>, char>::format(static_cast<std::underlying_type_t<E>>(value), ctx);
    }
};
