#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "calib/pipeline/planar_intrinsics.h"

namespace calib::planar {

[[nodiscard]] auto distortion_to_json(const Eigen::VectorXd& coeffs) -> nlohmann::json;

[[nodiscard]] auto camera_matrix_to_json(const CameraMatrix& k) -> nlohmann::json;

[[nodiscard]] auto compute_global_rms(const CalibrationOutputs& out) -> double;

[[nodiscard]] auto build_planar_intrinsics_report(
    const PlanarCalibrationConfig& cfg, const CameraConfig& cam_cfg,
    const PlanarDetections& detections, const CalibrationOutputs& outputs,
    const std::filesystem::path& features_path) -> nlohmann::json;

}  // namespace calib::planar
