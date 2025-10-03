#pragma once

// std
#include <string>
#include <unordered_map>
#include <vector>

#include "calib/io/serialization.h"
#include "calib/estimation/optim/extrinsics.h"
#include "calib/pipeline/facades/intrinsics.h"

namespace calib::pipeline {

struct StereoViewSelection final {
    std::string reference_image;
    std::string target_image;
};

struct StereoPairConfig final {
    std::string pair_id;
    std::string reference_sensor;
    std::string target_sensor;
    std::vector<StereoViewSelection> views;
    ExtrinsicOptions options;

    StereoPairConfig();
};

struct StereoCalibrationConfig final {
    std::vector<StereoPairConfig> pairs;
};

struct StereoCalibrationViewSummary final {
    std::string reference_image;
    std::string target_image;
    std::size_t reference_points = 0;
    std::size_t target_points = 0;
    std::string status;
};

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
                                 const PlanarDetections& reference_detections,
                                 const PlanarDetections& target_detections,
                                 const IntrinsicCalibrationOutputs& reference_intrinsics,
                                 const IntrinsicCalibrationOutputs& target_intrinsics) const
        -> StereoCalibrationRunResult;
};

// ---- Multicam generalization ----

struct MultiCameraViewSelection final {
    // Map sensor_id -> image filename for this view
    std::unordered_map<std::string, std::string> images;
};

struct MultiCameraRigConfig final {
    std::string rig_id;
    std::vector<std::string> sensors;  // order defines camera index mapping
    std::vector<MultiCameraViewSelection> views;
    ExtrinsicOptions options;
};

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
        const std::unordered_map<std::string, PlanarDetections>& detections_by_sensor,
        const std::unordered_map<std::string, IntrinsicCalibrationOutputs>& intrinsics_by_sensor)
        const -> MultiCameraCalibrationRunResult;
};

}  // namespace calib::pipeline
