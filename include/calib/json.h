#pragma once

#include <array>
#include <boost/pfr.hpp>
#include <nlohmann/json.hpp>
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

    boost::pfr::for_each_field(value, [&j, idx = 0](const auto& field) mutable {
        // Use a string representation of the index as the key
        j["field_" + std::to_string(idx++)] = field;
    });
}

// Populate aggregate types from JSON using field reflection.
template <Aggregate T>
void from_json(const nlohmann::json& j, T& value) {
    boost::pfr::for_each_field(value, [&j, idx = 0](auto& field) mutable {
        // Use a string representation of the index as the key
        j.at("field_" + std::to_string(idx++)).get_to(field);
    });
}

}  // namespace calib
