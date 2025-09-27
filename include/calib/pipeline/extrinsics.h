#pragma once

// std
#include <string>
#include <unordered_map>
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

// ---- Multicam generalization ----

struct MultiCameraViewSelection final {
    // Map sensor_id -> image filename for this view
    std::unordered_map<std::string, std::string> images;
};

void to_json(nlohmann::json& j, const MultiCameraViewSelection& view);
void from_json(const nlohmann::json& j, MultiCameraViewSelection& view);

struct MultiCameraRigConfig final {
    std::string rig_id;
    std::vector<std::string> sensors;  // order defines camera index mapping
    std::vector<MultiCameraViewSelection> views;
    ExtrinsicOptions options;
};

void to_json(nlohmann::json& j, const MultiCameraRigConfig& cfg);
void from_json(const nlohmann::json& j, MultiCameraRigConfig& cfg);

struct MultiCameraCalibrationRunResult final {
    bool success = false;
    std::size_t requested_views = 0;
    std::size_t used_views = 0;
    std::vector<std::string> sensors;  // same order as used in estimation
    ExtrinsicPoses initial_guess;
    ExtrinsicOptimizationResult<PinholeCamera<BrownConradyd>> optimization;
};

class MultiCameraCalibrationFacade {
  public:
    [[nodiscard]] auto calibrate(
        const MultiCameraRigConfig& cfg,
        const std::unordered_map<std::string, planar::PlanarDetections>& detections_by_sensor,
        const std::unordered_map<std::string, planar::CalibrationRunResult>& intrinsics_by_sensor)
        const -> MultiCameraCalibrationRunResult;
};

}  // namespace calib::pipeline
