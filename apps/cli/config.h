/** @brief Calib app config */

// std
#include <string>

// nlohmann
#include <nlohmann/json.hpp>

#pragma once

struct AppConfig final {
    std::string task;
    std::string output;
    std::string input_path;
};

inline void to_json(nlohmann::json& j, const AppConfig& cfg) {
    j = {{"task", cfg.task}, {"output", cfg.output}, {"input_path", cfg.input_path}};
}

inline void from_json(const nlohmann::json& j, AppConfig& cfg) {
    j.at("task").get_to(cfg.task);
    j.at("output").get_to(cfg.output);
    j.at("input_path").get_to(cfg.input_path);
}
