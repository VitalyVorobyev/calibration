#pragma once

// std
#include <vector>

// third-party
#include <nlohmann/json.hpp>

#include "calib/datasets/planar.h"

namespace calib::pipeline {

/**
 * @brief Aggregated dataset consumed by the calibration pipeline.
 */
struct CalibrationDataset {
    int schema_version{1};
    nlohmann::json metadata;
    std::vector<planar::PlanarDetections> planar_cameras;
    nlohmann::json raw_json;  ///< Original dataset payload for downstream consumers.
};

}  // namespace calib::pipeline
