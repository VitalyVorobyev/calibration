/** @brief Optimization of camera intrinsics parameters */

#pragma once

// std
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/planarpose.h"
#include "calib/camera.h"
#include "calib/cameramodel.h"

#include "calib/optimize.h"

namespace calib {

// Estimate camera intrinsics (fx, fy, cx, cy[, skew]) by solving a linear
// least-squares system that ignores lens distortion.  The input
// observations contain normalized coordinates (x,y) for an undistorted
// point and the observed pixel coordinates (u,v).  The function returns
// an optional CameraMatrix: std::nullopt is returned if there are not
// enough observations or the linear system is degenerate.
std::optional<CameraMatrix> estimate_intrinsics_linear(
    const std::vector<Observation<double>>& obs,
    std::optional<CalibrationBounds> bounds = std::nullopt,
    bool use_skew = false);

// Improved linear initialization that alternates between estimating
// distortion coefficients (fit_distortion) and re-solving for camera
// intrinsics (estimate_intrinsics_linear).  Returns std::nullopt if the
// initial linear estimation fails.  The number of radial distortion
// coefficients and the number of refinement iterations can be specified.
std::optional<Camera<BrownConradyd>> estimate_intrinsics_linear_iterative(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    int max_iterations = 5,
    bool use_skew = false);

struct IntrinsicsOptions final : public OptimOptions {
    int num_radial = 2;  ///< Number of radial distortion coefficients
    bool optimize_skew = false;  ///< Estimate skew parameter
    std::optional<CalibrationBounds> bounds = std::nullopt;  ///< Parameter bounds
};

template<camera_model CameraT>
struct IntrinsicsOptimizationResult final : public OptimResult {
    CameraT camera;                      ///< Estimated camera parameters
    std::vector<Eigen::Affine3d> c_T_t;  ///< Estimated pose of each view
    std::vector<double> view_errors;     ///< Per-view reprojection errors
};

IntrinsicsOptimizationResult<Camera<BrownConradyd>> optimize_intrinsics_semidlt(
    const std::vector<PlanarView>& views,
    const CameraMatrix& initial_guess,
    const IntrinsicsOptions& opts = {});

template<camera_model CameraT>
IntrinsicsOptimizationResult<CameraT> optimize_intrinsics(
    const std::vector<PlanarView>& views,
    const CameraT& init_camera,
    std::vector<Eigen::Affine3d> init_c_T_t,
    const IntrinsicsOptions& opts = {});

}  // namespace calib
