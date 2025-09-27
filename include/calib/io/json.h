#pragma once

#include <array>
#include <boost/pfr.hpp>
#include <boost/pfr/core_name.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <type_traits>

namespace calib {

// Concept for aggregate types that can be reflected by Boost.PFR.
template <typename T>
concept Aggregate = std::is_aggregate_v<T> && std::is_class_v<T>;

// Convert aggregate types to JSON using field reflection.
template <Aggregate T>
void to_json(nlohmann::json& j, const T& value) {
    j = nlohmann::json::object();

#if defined(BOOST_PFR_CORE_NAME_ENABLED)
    constexpr auto names = boost::pfr::names_as_array<T>();
    boost::pfr::for_each_field(value, [&j, &names, idx = 0](const auto& field) mutable {
        // Prefer real field names when available; fallback to indexed keys
        const std::string key = std::string(names[idx].empty() ? std::string_view{} : names[idx]);
        const std::string idx_key = "field_" + std::to_string(idx);
        j[!key.empty() ? key : idx_key] = field;
        ++idx;
    });
#else
    boost::pfr::for_each_field(value, [&j, idx = 0](const auto& field) mutable {
        j["field_" + std::to_string(idx++)] = field;
    });
#endif
}

// Populate aggregate types from JSON using field reflection.
template <Aggregate T>
void from_json(const nlohmann::json& j, T& value) {
#if defined(BOOST_PFR_CORE_NAME_ENABLED)
    constexpr auto names = boost::pfr::names_as_array<T>();
    boost::pfr::for_each_field(value, [&j, &names, idx = 0](auto& field) mutable {
        const std::string name_key = std::string(names[idx].empty() ? std::string_view{} : names[idx]);
        const std::string idx_key = "field_" + std::to_string(idx);
        // Try by name first (new format), then fallback to legacy indexed keys
        if (!name_key.empty() && j.contains(name_key)) {
            j.at(name_key).get_to(field);
        } else if (j.contains(idx_key)) {
            j.at(idx_key).get_to(field);
        } else {
            // Keep prior semantics: let at() throw with the name key if neither exists
            j.at(!name_key.empty() ? name_key : idx_key).get_to(field);
        }
        ++idx;
    });
#else
    boost::pfr::for_each_field(value, [&j, idx = 0](auto& field) mutable {
        j.at("field_" + std::to_string(idx++)).get_to(field);
    });
#endif
}

}  // namespace calib
