#pragma once

// std
#include <set>
#include <string>
#include <vector>

#include "calib/io/serialization.h"

namespace calib::pipeline {

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

/**
 * @brief Aggregated dataset consumed by the calibration pipeline.
 */
struct CalibrationDataset final {
    int schema_version{1};
    nlohmann::json metadata;
    std::vector<PlanarDetections> planar_cameras;
    nlohmann::json raw_json;  ///< Original dataset payload for downstream consumers.
};

}  // namespace calib::pipeline
