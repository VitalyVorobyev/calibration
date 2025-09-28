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

// third-party
#include <nlohmann/json.hpp>

#include "calib/estimation/optim/bundle.h"
#include "calib/estimation/optim/handeye.h"
#include "calib/io/serialization.h"

namespace calib::pipeline {

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
    HandeyeOptions options;
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

inline void to_json(nlohmann::json& j, const HandEyeObservationConfig& cfg) {
    j = nlohmann::json{{"images", cfg.images}, {"base_se3_gripper", cfg.base_se3_gripper}};
    if (!cfg.view_id.empty()) {
        j["id"] = cfg.view_id;
    }
}

inline void from_json(const nlohmann::json& j, HandEyeObservationConfig& cfg) {
    cfg.view_id = j.value("id", std::string{});
    if (j.contains("base_se3_gripper")) {
        cfg.base_se3_gripper = j.at("base_se3_gripper").get<Eigen::Isometry3d>();
    } else if (j.contains("b_T_g")) {
        cfg.base_se3_gripper = j.at("b_T_g").get<Eigen::Isometry3d>();
    } else {
        throw std::runtime_error("HandEyeObservationConfig: missing 'base_se3_gripper'");
    }
    if (j.contains("images")) {
        cfg.images = j.at("images").get<std::unordered_map<std::string, std::string>>();
    } else {
        cfg.images.clear();
    }
}

inline void to_json(nlohmann::json& j, const HandEyeRigConfig& cfg) {
    j = nlohmann::json::object();
    j["rig_id"] = cfg.rig_id;
    j["sensors"] = cfg.sensors;
    j["observations"] = cfg.observations;
    nlohmann::json options_json;
    to_json(options_json, cfg.options);
    j["options"] = std::move(options_json);
    j["min_angle_deg"] = cfg.min_angle_deg;
}

inline void from_json(const nlohmann::json& j, HandEyeRigConfig& cfg) {
    cfg.rig_id = j.value("rig_id", std::string{});
    j.at("sensors").get_to(cfg.sensors);
    cfg.observations.clear();
    if (j.contains("observations")) {
        j.at("observations").get_to(cfg.observations);
    }
    if (j.contains("options")) {
        j.at("options").get_to(cfg.options);
    }
    cfg.min_angle_deg = j.value("min_angle_deg", cfg.min_angle_deg);
    if (cfg.rig_id.empty() && !cfg.sensors.empty()) {
        cfg.rig_id = cfg.sensors.front();
    }
}

inline void to_json(nlohmann::json& j, const HandEyePipelineConfig& cfg) {
    j = nlohmann::json{{"rigs", cfg.rigs}};
}

inline void from_json(const nlohmann::json& j, HandEyePipelineConfig& cfg) {
    cfg.rigs.clear();
    if (j.contains("rigs")) {
        j.at("rigs").get_to(cfg.rigs);
    }
}

inline void to_json(nlohmann::json& j, const BundleRigConfig& cfg) {
    j = nlohmann::json::object();
    j["rig_id"] = cfg.rig_id;
    j["sensors"] = cfg.sensors;
    j["observations"] = cfg.observations;
    nlohmann::json options_json;
    to_json(options_json, cfg.options);
    j["options"] = std::move(options_json);
    j["min_angle_deg"] = cfg.min_angle_deg;
    if (cfg.initial_target.has_value()) {
        j["initial_target"] = cfg.initial_target.value();
    }
}

inline void from_json(const nlohmann::json& j, BundleRigConfig& cfg) {
    cfg.rig_id = j.value("rig_id", std::string{});
    j.at("sensors").get_to(cfg.sensors);
    cfg.observations.clear();
    if (j.contains("observations")) {
        j.at("observations").get_to(cfg.observations);
    }
    if (j.contains("options")) {
        j.at("options").get_to(cfg.options);
    }
    cfg.min_angle_deg = j.value("min_angle_deg", cfg.min_angle_deg);
    if (j.contains("initial_target")) {
        cfg.initial_target = j.at("initial_target").get<Eigen::Isometry3d>();
    } else {
        cfg.initial_target.reset();
    }
    if (cfg.rig_id.empty() && !cfg.sensors.empty()) {
        cfg.rig_id = cfg.sensors.front();
    }
}

inline void to_json(nlohmann::json& j, const BundlePipelineConfig& cfg) {
    j = nlohmann::json{{"rigs", cfg.rigs}};
}

inline void from_json(const nlohmann::json& j, BundlePipelineConfig& cfg) {
    cfg.rigs.clear();
    if (j.contains("rigs")) {
        j.at("rigs").get_to(cfg.rigs);
    }
}

}  // namespace calib::pipeline
