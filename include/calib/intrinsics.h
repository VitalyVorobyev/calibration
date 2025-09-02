/** @brief Optimization of camera intrinsics parameters */

#pragma once

// std
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/camera.h"
#include "calib/cameramodel.h"
#include "calib/optimize.h"
#include "calib/planarpose.h"

namespace calib {

// Estimate camera intrinsics (fx, fy, cx, cy[, skew]) by solving a linear
// least-squares system that ignores lens distortion.  The input
// observations contain normalized coordinates (x,y) for an undistorted
// point and the observed pixel coordinates (u,v).  The function returns
// an optional CameraMatrix: std::nullopt is returned if there are not
// enough observations or the linear system is degenerate.
auto estimate_intrinsics_linear(const std::vector<Observation<double>>& observations,
                                std::optional<CalibrationBounds> bounds = std::nullopt,
                                bool use_skew = false) -> std::optional<CameraMatrix>;

// Improved linear initialization that alternates between estimating
// distortion coefficients (fit_distortion) and re-solving for camera
// intrinsics (estimate_intrinsics_linear).  Returns std::nullopt if the
// initial linear estimation fails.  The number of radial distortion
// coefficients and the number of refinement iterations can be specified.
constexpr int kDefaultMaxIterations = 5;
auto estimate_intrinsics_linear_iterative(
    const std::vector<Observation<double>>& observations, int num_radial,
    int max_iterations = kDefaultMaxIterations,
    bool use_skew = false) -> std::optional<Camera<BrownConradyd>>;

struct IntrinsicsOptions final : public OptimOptions {
    int num_radial = 2;          ///< Number of radial distortion coefficients
    bool optimize_skew = false;  ///< Estimate skew parameter
    std::optional<CalibrationBounds> bounds = std::nullopt;  ///< Parameter bounds
};

template <camera_model CameraT>
struct IntrinsicsOptimizationResult final : public OptimResult {
    CameraT camera;                      ///< Estimated camera parameters
    std::vector<Eigen::Isometry3d> c_se3_t;  ///< Estimated pose of each view
    std::vector<double> view_errors;     ///< Per-view reprojection errors
};

auto optimize_intrinsics_semidlt(
    const std::vector<PlanarView>& views, const CameraMatrix& initial_guess,
    const IntrinsicsOptions& opts = {}) -> IntrinsicsOptimizationResult<Camera<BrownConradyd>>;

template <camera_model CameraT>
auto optimize_intrinsics(const std::vector<PlanarView>& views, const CameraT& init_camera,
                         std::vector<Eigen::Isometry3d> init_c_se3_t,
                         const IntrinsicsOptions& opts = {})
    -> IntrinsicsOptimizationResult<CameraT>;

}  // namespace calib
