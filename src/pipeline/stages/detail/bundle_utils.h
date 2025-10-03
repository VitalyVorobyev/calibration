#pragma once

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "calib/estimation/optim/bundle.h"
#include "calib/estimation/optim/handeye.h"
#include "calib/pipeline/handeye.h"
#include "planar_utils.h"

namespace calib::pipeline::detail {

struct BundleSensorSetup final {
    std::vector<PinholeCamera<BrownConradyd>> cameras;
    std::unordered_map<std::string, std::size_t> sensor_to_index;
    std::vector<std::string> missing_sensors;
};

auto collect_bundle_sensor_setup(
    const BundleRigConfig& rig,
    const std::unordered_map<std::string, planar::CalibrationRunResult>& intrinsics)
    -> BundleSensorSetup;

const std::vector<HandEyeObservationConfig>* select_bundle_observations(
    const BundleRigConfig& rig, const HandEyePipelineConfig* handeye_cfg);

struct SensorAccumulator final {
    std::vector<Eigen::Isometry3d> base;
    std::vector<Eigen::Isometry3d> cam;
};

struct BundleViewProcessingResult final {
    nlohmann::json views;
    std::vector<BundleObservation> observations;
    std::vector<SensorAccumulator> accumulators;
    std::size_t used_view_count{0};
};

auto collect_bundle_observations(
    const std::vector<HandEyeObservationConfig>& observation_config,
    const std::vector<std::string>& sensors,
    const std::unordered_map<std::string, std::size_t>& sensor_to_index,
    const std::unordered_map<std::string, SensorDetectionsIndex>& sensor_index,
    const std::unordered_map<std::string, planar::CalibrationRunResult>& intrinsics)
    -> BundleViewProcessingResult;

struct HandeyeInitializationResult final {
    std::vector<Eigen::Isometry3d> transforms;
    nlohmann::json report;
    bool failed{false};
};

auto compute_handeye_initialization(
    const BundleRigConfig& rig,
    const std::unordered_map<std::string, std::unordered_map<std::string, HandeyeResult>>&
        handeye_results,
    const std::vector<SensorAccumulator>& accumulators) -> HandeyeInitializationResult;

struct TargetInitializationResult final {
    Eigen::Isometry3d pose{Eigen::Isometry3d::Identity()};
    std::string source;
};

auto choose_initial_target(const BundleRigConfig& rig,
                           const std::vector<SensorAccumulator>& accumulators,
                           const std::vector<Eigen::Isometry3d>& init_g_se3_c)
    -> TargetInitializationResult;

}  // namespace calib::pipeline::detail
