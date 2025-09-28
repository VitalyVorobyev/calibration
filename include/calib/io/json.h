#pragma once

#include <array>
#include <boost/pfr/core.hpp>

#if defined(__has_include)
#if __has_include(<boost/pfr/core_name.hpp>)
#include <boost/pfr/core_name.hpp>
#define CALIB_HAS_PFR_CORE_NAME 1
#endif
#endif

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace calib {

namespace detail {
template <class T>
struct is_optional : std::false_type {};
template <class U>
struct is_optional<std::optional<U>> : std::true_type {};
template <class T>
inline constexpr bool is_optional_v = is_optional<std::decay_t<T>>::value;

template <class Opt>
using optional_value_t = typename std::decay_t<Opt>::value_type;

inline std::string idx_key(std::size_t idx) {
    using std::to_string;
    return "field_" + to_string(idx);
}
}  // namespace detail

// ----------------------------
// Aggregate → JSON
// ----------------------------
template <class T, std::enable_if_t<std::is_aggregate_v<T>, int> = 0>
void to_json(nlohmann::json& j, const T& value) {
    j = nlohmann::json::object();

#if CALIB_HAS_PFR_CORE_NAME
    constexpr auto names = boost::pfr::names_as_array<T>();
    std::size_t idx = 0;

    boost::pfr::for_each_field(value, [&](auto const& field) mutable {
        const std::string i_key = detail::idx_key(idx);
        const std::string n_key = std::string(names[idx]);  // ← convert sv→string
        const bool have_named = !n_key.empty();

        if constexpr (detail::is_optional_v<decltype(field)>) {
            if (field) {
                j[i_key] = *field;
                if (have_named) j[n_key] = *field;
            }
        } else {
            j[i_key] = field;
            if (have_named) j[n_key] = field;
        }
        ++idx;
    });
#else
    std::size_t idx = 0;
    boost::pfr::for_each_field(value, [&](auto const& field) mutable {
        const std::string i_key = detail::idx_key(idx++);
        if constexpr (detail::is_optional_v<decltype(field)>) {
            if (field) j[i_key] = *field;  // omit when nullopt
        } else {
            j[i_key] = field;
        }
    });
#endif
}

// ----------------------------
// JSON → Aggregate
// ----------------------------
template <class T, std::enable_if_t<std::is_aggregate_v<T>, int> = 0>
void from_json(const nlohmann::json& j, T& value) {
#if CALIB_HAS_PFR_CORE_NAME
    constexpr auto names = boost::pfr::names_as_array<T>();
    std::size_t idx = 0;

    boost::pfr::for_each_field(value, [&](auto& field) mutable {
        const std::string i_key = detail::idx_key(idx);
        const std::string n_key = std::string(names[idx]);  // ← convert sv→string
        const bool have_named = !n_key.empty();

        auto read_optional = [&](auto& opt) {
            using Inner = detail::optional_value_t<decltype(opt)>;

            // Prefer named key, then index key
            const nlohmann::json* slot = nullptr;
            if (have_named) {
                if (auto it = j.find(n_key); it != j.end()) slot = &*it;
            }
            if (!slot) {
                if (auto it = j.find(i_key); it != j.end()) slot = &*it;
            }

            if (!slot || slot->is_null()) {
                opt.reset();
            } else {
                opt = slot->get<Inner>();
            }
        };

        if constexpr (detail::is_optional_v<decltype(field)>) {
            read_optional(field);
        } else {
            if (have_named && j.contains(n_key)) {
                j.at(n_key).get_to(field);
            } else if (j.contains(i_key)) {
                j.at(i_key).get_to(field);
            } else {
                // Throw with the most user-friendly key we can show
                j.at(have_named ? n_key : i_key).get_to(field);
            }
        }
        ++idx;
    });
#else
    std::size_t idx = 0;
    boost::pfr::for_each_field(value, [&](auto& field) mutable {
        const std::string i_key = detail::idx_key(idx++);

        if constexpr (detail::is_optional_v<decltype(field)>) {
            using Inner = detail::optional_value_t<decltype(field)>;
            if (auto it = j.find(i_key); it == j.end() || it->is_null()) {
                field.reset();
            } else {
                field = it->get<Inner>();
            }
        } else {
            j.at(i_key).get_to(field);  // required
        }
    });
#endif
}

}  // namespace calib