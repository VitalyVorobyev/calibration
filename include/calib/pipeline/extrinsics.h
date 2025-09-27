#pragma once

// std
#include <string>
#include <unordered_map>
#include <vector>

// third-party
#include <nlohmann/json.hpp>

#include "calib/datasets/planar.h"
#include "calib/estimation/optim/extrinsics.h"
#include "calib/io/serialization.h"
#include "calib/pipeline/planar_intrinsics.h"

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

struct MultiCameraRigConfig final {
    std::string rig_id;
    std::vector<std::string> sensors;  // order defines camera index mapping
    std::vector<MultiCameraViewSelection> views;
    ExtrinsicOptions options;
};

inline void to_json(nlohmann::json& j, const MultiCameraViewSelection& view) { j = view.images; }

inline void from_json(const nlohmann::json& j, MultiCameraViewSelection& view) {
    view.images = j.get<std::unordered_map<std::string, std::string>>();
}

inline void to_json(nlohmann::json& j, const MultiCameraRigConfig& cfg) {
    nlohmann::json options_json;
    to_json(options_json, cfg.options);
    j = {{"rig_id", cfg.rig_id},
         {"sensors", cfg.sensors},
         {"views", cfg.views},
         {"options", options_json}};
}

inline void from_json(const nlohmann::json& j, MultiCameraRigConfig& cfg) {
    cfg.rig_id = j.value("rig_id", std::string{});
    j.at("sensors").get_to(cfg.sensors);
    if (cfg.rig_id.empty() && !cfg.sensors.empty()) cfg.rig_id = cfg.sensors.front();
    cfg.views.clear();
    if (j.contains("views")) j.at("views").get_to(cfg.views);
    if (j.contains("options")) j.at("options").get_to(cfg.options);
}

inline void to_json(nlohmann::json& j, const StereoViewSelection& view) {
    j = {{"reference_image", view.reference_image}, {"target_image", view.target_image}};
}

inline void from_json(const nlohmann::json& j, StereoViewSelection& view) {
    j.at("reference_image").get_to(view.reference_image);
    j.at("target_image").get_to(view.target_image);
}

inline void to_json(nlohmann::json& j, const StereoPairConfig& cfg) {
    j = {{"pair_id", cfg.pair_id},
         {"reference_sensor", cfg.reference_sensor},
         {"target_sensor", cfg.target_sensor},
         {"views", cfg.views},
         {"options", cfg.options}};
}

inline void from_json(const nlohmann::json& j, StereoPairConfig& cfg) {
    cfg.pair_id = j.value("pair_id", std::string{});
    j.at("reference_sensor").get_to(cfg.reference_sensor);
    j.at("target_sensor").get_to(cfg.target_sensor);
    cfg.views.clear();
    if (j.contains("views")) j.at("views").get_to(cfg.views);
    if (j.contains("options")) j.at("options").get_to(cfg.options);
    if (cfg.pair_id.empty()) cfg.pair_id = cfg.reference_sensor + "_" + cfg.target_sensor;
}

inline void to_json(nlohmann::json& j, const StereoCalibrationConfig& cfg) {
    j = {{"pairs", cfg.pairs}};
}

inline void from_json(const nlohmann::json& j, StereoCalibrationConfig& cfg) {
    cfg.pairs.clear();
    if (j.contains("pairs")) j.at("pairs").get_to(cfg.pairs);
}

inline void to_json(nlohmann::json& j, const StereoCalibrationViewSummary& summary) {
    j = {{"reference_image", summary.reference_image},
         {"target_image", summary.target_image},
         {"reference_points", summary.reference_points},
         {"target_points", summary.target_points},
         {"status", summary.status}};
}

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
