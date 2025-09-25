#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace calib::planar {

struct PlanarTargetPoint {
    double x = 0.0;
    double y = 0.0;
    int id = -1;
    double local_x = 0.0;
    double local_y = 0.0;
    double local_z = 0.0;
};

struct PlanarImageDetections {
    std::string file;
    int count = 0;
    std::vector<PlanarTargetPoint> points;
};

struct PlanarDetections {
    std::string image_directory;
    std::string feature_type;
    std::string algo_version;
    std::string params_hash;
    std::string sensor_id;
    std::set<std::string> tags;
    nlohmann::json metadata = nlohmann::json::object();
    std::filesystem::path source_file;
    std::vector<PlanarImageDetections> images;
};

[[nodiscard]] auto validate_planar_dataset(const nlohmann::json& dataset,
                                           std::string* error_message = nullptr) -> bool;

[[nodiscard]] auto convert_legacy_planar_features(const nlohmann::json& legacy,
                                                  const std::string& sensor_id_hint = "cam0")
    -> nlohmann::json;

[[nodiscard]] auto load_planar_dataset(const std::filesystem::path& path,
                                       std::optional<std::string> sensor_filter = std::nullopt)
    -> PlanarDetections;

}  // namespace calib::planar
