/** @brief Optimization of camera intrinsics parameters */

#pragma once

// std
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/distortion.h"  // Observation
#include "calib/cameramatrix.h"
#include "calib/camera.h"

namespace calib {

struct IntrinsicOptimizationResult {
    Camera<DualDistortion> camera;
    Eigen::Matrix4d covariance;  // Covariance matrix of intrinsics
    double reprojection_error;   // Reprojection error after optimization (pix)
    std::string summary;         // Summary of optimization results
};

// Result of an iterative linear initialization that alternates between
// estimating camera intrinsics and lens distortion parameters.
struct LinearInitResult {
    Camera<DualDistortion> camera;
};

// Estimate camera intrinsics (fx, fy, cx, cy) by solving a linear
// least-squares system that ignores lens distortion.  The input
// observations contain normalized coordinates (x,y) for an undistorted
// point and the observed pixel coordinates (u,v).  The function returns
// an optional CameraMatrix: std::nullopt is returned if there are not
// enough observations or the linear system is degenerate.
std::optional<CameraMatrix> estimate_intrinsics_linear(
    const std::vector<Observation<double>>& obs,
    std::optional<CalibrationBounds> bounds = std::nullopt);

// Improved linear initialization that alternates between estimating
// distortion coefficients (fit_distortion) and re-solving for camera
// intrinsics (estimate_intrinsics_linear).  Returns std::nullopt if the
// initial linear estimation fails.  The number of radial distortion
// coefficients and the number of refinement iterations can be specified.
std::optional<LinearInitResult> estimate_intrinsics_linear_iterative(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    int max_iterations = 5);

IntrinsicOptimizationResult optimize_intrinsics(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verb=false,
    std::optional<CalibrationBounds> bounds = std::nullopt
);

}  // namespace calib
