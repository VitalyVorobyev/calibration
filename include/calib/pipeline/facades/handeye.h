/**
 * @file handeye.h
 * @brief Pipeline configuration helpers for hand-eye and bundle adjustment stages.
 */

#pragma once

// std
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/estimation/optim/bundle.h"
#include "calib/estimation/optim/handeye.h"
#include "calib/io/serialization.h"

namespace calib::pipeline {

using calib::from_json;
using calib::to_json;

/**
 * @brief Per-observation specification consumed by the hand-eye pipeline stage.
 *
 * Each entry binds a robot pose (`base_se3_gripper`) to captured image detections.
 * The `images` map must contain filenames present in the planar detection dataset
 * for every sensor participating in the calibration rig.
 */
struct HandEyeObservationConfig final {
    std::string view_id;
    Eigen::Isometry3d base_se3_gripper{Eigen::Isometry3d::Identity()};
    std::unordered_map<std::string, std::string> images;
};

/**
 * @brief Hand-eye rig definition used to configure the calibration stage.
 */
struct HandEyeRigConfig final {
    std::string rig_id;
    std::vector<std::string> sensors;
    std::vector<HandEyeObservationConfig> observations;
    OptimOptions options;
    double min_angle_deg{1.0};
};

struct HandEyePipelineConfig final {
    std::vector<HandEyeRigConfig> rigs;
};

/**
 * @brief Bundle-adjustment rig configuration extending the hand-eye observations.
 */
struct BundleRigConfig final {
    std::string rig_id;
    std::vector<std::string> sensors;
    std::vector<HandEyeObservationConfig> observations;
    BundleOptions options;
    double min_angle_deg{1.0};
    std::optional<Eigen::Isometry3d> initial_target;
};

struct BundlePipelineConfig final {
    std::vector<BundleRigConfig> rigs;
};

static_assert(serializable_aggregate<HandEyeObservationConfig>);
static_assert(serializable_aggregate<HandEyeRigConfig>);
static_assert(serializable_aggregate<HandEyePipelineConfig>);
static_assert(serializable_aggregate<BundleRigConfig>);
static_assert(serializable_aggregate<BundlePipelineConfig>);

}  // namespace calib::pipeline
