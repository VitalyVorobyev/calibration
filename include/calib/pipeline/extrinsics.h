#pragma once

// std
#include <string>
#include <vector>

// third-party
#include <nlohmann/json.hpp>

#include "calib/datasets/planar.h"
#include "calib/estimation/extrinsics.h"
#include "calib/pipeline/planar_intrinsics.h"

namespace calib::pipeline {

struct StereoViewSelection final {
    std::string reference_image;
    std::string target_image;
};

void to_json(nlohmann::json& j, const StereoViewSelection& view);
void from_json(const nlohmann::json& j, StereoViewSelection& view);

struct StereoPairConfig final {
    std::string pair_id;
    std::string reference_sensor;
    std::string target_sensor;
    std::vector<StereoViewSelection> views;
    ExtrinsicOptions options;

    StereoPairConfig();
};

void to_json(nlohmann::json& j, const StereoPairConfig& cfg);
void from_json(const nlohmann::json& j, StereoPairConfig& cfg);

struct StereoCalibrationConfig final {
    std::vector<StereoPairConfig> pairs;
};

void to_json(nlohmann::json& j, const StereoCalibrationConfig& cfg);
void from_json(const nlohmann::json& j, StereoCalibrationConfig& cfg);

struct StereoCalibrationViewSummary final {
    std::string reference_image;
    std::string target_image;
    std::size_t reference_points = 0;
    std::size_t target_points = 0;
    std::string status;
};

void to_json(nlohmann::json& j, const StereoCalibrationViewSummary& summary);

struct StereoCalibrationRunResult final {
    bool success = false;
    std::size_t requested_views = 0;
    std::size_t used_views = 0;
    std::vector<StereoCalibrationViewSummary> view_summaries;
    ExtrinsicPoses initial_guess;
    ExtrinsicOptimizationResult<PinholeCamera<BrownConradyd>> optimization;
};

class StereoCalibrationFacade {
  public:
    [[nodiscard]] auto calibrate(const StereoPairConfig& cfg,
                                 const planar::PlanarDetections& reference_detections,
                                 const planar::PlanarDetections& target_detections,
                                 const planar::CalibrationRunResult& reference_intrinsics,
                                 const planar::CalibrationRunResult& target_intrinsics) const
        -> StereoCalibrationRunResult;
};

}  // namespace calib::pipeline
