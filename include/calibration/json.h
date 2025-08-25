#pragma once

#include <type_traits>

#include <boost/pfr.hpp>
#include <nlohmann/json.hpp>

namespace vitavision {

// Concept for aggregate types that can be reflected by Boost.PFR.
template <typename T>
concept Aggregate = std::is_aggregate_v<T> && std::is_class_v<T>;

// Convert aggregate types to JSON using field reflection.
template <Aggregate T>
void to_json(nlohmann::json& j, const T& value) {
    j = nlohmann::json::object();
    constexpr auto names = boost::pfr::names_as_array<T>();
    boost::pfr::for_each_field(value, [&](const auto& field, std::size_t i) {
        j[names[i]] = field;
    });
}

// Populate aggregate types from JSON using field reflection.
template <Aggregate T>
void from_json(const nlohmann::json& j, T& value) {
    constexpr auto names = boost::pfr::names_as_array<T>();
    boost::pfr::for_each_field(value, [&](auto& field, std::size_t i) {
        j.at(names[i]).get_to(field);
    });
}

}  // namespace vitavision

