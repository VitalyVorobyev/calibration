#pragma once

// std
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "calib/io/json.h"

namespace calib::planar {

struct PlanarTargetPoint final {
    double x = 0.0;
    double y = 0.0;
    int id = -1;
    double local_x = 0.0;
    double local_y = 0.0;
    double local_z = 0.0;
};

struct PlanarImageDetections final {
    std::string file;
    std::vector<PlanarTargetPoint> points;
};

struct PlanarDetections final {
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

}  // namespace calib::planar
