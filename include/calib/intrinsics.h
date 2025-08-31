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
#include "calib/planarpose.h"  // PlanarView

namespace calib {

// Result of an iterative linear initialization that alternates between
// estimating camera intrinsics and lens distortion parameters.
struct LinearInitResult {
    Camera<DualDistortion> camera;
};

struct IntrinsicsOptions final {
    int num_radial = 2;  ///< Number of radial distortion coefficients
    bool optimize_skew = false;  ///< Estimate skew parameter
    std::optional<CalibrationBounds> bounds = std::nullopt;  ///< Parameter bounds
    bool verbose = false;  ///< Verbose solver output
};

struct IntrinsicsResult final {
    Camera<DualDistortion> camera;              ///< Estimated camera parameters
    std::vector<Eigen::Affine3d> poses;         ///< Estimated pose of each view
    Eigen::MatrixXd covariance;                ///< Covariance of intrinsics and poses
    std::vector<double> view_errors;           ///< Per-view reprojection errors
    double reprojection_error = 0.0;           ///< Overall reprojection RMSE
    std::string summary;                       ///< Solver brief report
};

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
std::optional<LinearInitResult> estimate_intrinsics_linear_iterative(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    int max_iterations = 5,
    bool use_skew = false);

IntrinsicsResult optimize_intrinsics(
    const std::vector<PlanarView>& views,
    const CameraMatrix& initial_guess,
    const IntrinsicsOptions& opts = {});

}  // namespace calib
