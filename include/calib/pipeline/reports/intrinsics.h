#pragma once

#include <filesystem>

#include "calib/io/json.h"
#include "calib/models/camera_matrix.h"
#include "calib/pipeline/facades/intrinsics.h"

namespace calib::pipeline {

using calib::from_json;
using calib::to_json;

struct InitialGuessWarningCounts final {
    std::size_t invalid_camera_matrix = 0;
    std::size_t homography_decomposition_failures = 0;
};

struct InitialGuessReport final {
    CameraMatrix intrinsics;
    std::vector<std::size_t> used_view_indices;
    InitialGuessWarningCounts warning_counts;
};

struct PlanarViewReport final {
    std::string source_image;
    std::size_t corner_count = 0;
    double rms_px = 0.0;
    bool used_in_linear_stage = false;
};

struct IntrinsicsResultReport final {
    CameraMatrix intrinsics;
    std::string distortion_model;
    std::vector<double> distortion_coefficients;
    double reprojection_rms_px = 0.0;
    std::vector<PlanarViewReport> per_view;
};

struct CameraReport final {
    std::string camera_id;
    std::string model;
    std::optional<std::array<int, 2>> image_size;
    InitialGuessReport initial_guess;
    IntrinsicsResultReport result;
};

struct CalibrationReport final {
    std::string type;
    std::string algorithm;
    IntrinsicCalibrationOptions options;
    nlohmann::json detector;
    std::vector<CameraReport> cameras;
};

[[nodiscard]] auto build_planar_intrinsics_report(const IntrinsicCalibrationConfig& cfg,
                                                  const CameraConfig& cam_cfg,
                                                  const PlanarDetections& detections,
                                                  const IntrinsicCalibrationOutputs& outputs)
    -> CalibrationReport;

static_assert(serializable_aggregate<CalibrationReport>);
static_assert(serializable_aggregate<CameraReport>);
static_assert(serializable_aggregate<InitialGuessReport>);
static_assert(serializable_aggregate<InitialGuessWarningCounts>);
static_assert(serializable_aggregate<IntrinsicsResultReport>);
static_assert(serializable_aggregate<PlanarViewReport>);

}  // namespace calib::pipeline
