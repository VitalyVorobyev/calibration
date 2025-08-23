/** @brief Optimization of camera intrinsics parameters */

#pragma once

// std
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calibration/distortion.h"  // Observation

namespace vitavision {

template<typename T>
struct CameraMatrix final {
    T fx, fy, cx, cy;

    Eigen::Matrix<T, 2, 1> normalize(const Eigen::Matrix<T, 2, 1>& pix) const {
        return {
            (pix.x() - T(cx)) / T(fx),
            (pix.y() - T(cy)) / T(fy)
        };
    }

    Eigen::Matrix<T, 2, 1> denormalize(const Eigen::Matrix<T, 2, 1>& xy) const {
        return {
            T(fx) * xy.x() + T(cx),
            T(fy) * xy.y() + T(cy)
        };
    }
};

// Bounds for intrinsics parameters used during calibration. The default
// values roughly correspond to a 1280x720 image.
struct CalibrationBounds {
    double fx_min = 100.0;
    double fx_max = 2000.0;
    double fy_min = 100.0;
    double fy_max = 2000.0;
    double cx_min = 10.0;
    double cx_max = 1280.0;
    double cy_min = 10.0;
    double cy_max = 720.0;
};

struct IntrinsicOptimizationResult {
    CameraMatrix<double> intrinsics;
    Eigen::VectorXd distortion;
    Eigen::Matrix4d covariance;  // Covariance matrix of intrinsics
    double reprojection_error;   // Reprojection error after optimization (pix)
    std::string summary;         // Summary of optimization results
};

// Result of an iterative linear initialization that alternates between
// estimating camera intrinsics and lens distortion parameters.
struct LinearInitResult {
    CameraMatrix<double> intrinsics;     // Estimated camera matrix
    Eigen::VectorXd distortion;  // Estimated distortion coefficients
};

// Estimate camera intrinsics (fx, fy, cx, cy) by solving a linear
// least-squares system that ignores lens distortion.  The input
// observations contain normalized coordinates (x,y) for an undistorted
// point and the observed pixel coordinates (u,v).  The function returns
// an optional CameraMatrix: std::nullopt is returned if there are not
// enough observations or the linear system is degenerate.
std::optional<CameraMatrix<double>> estimate_intrinsics_linear(
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
    const CameraMatrix<double>& initial_guess,
    bool verb=false,
    std::optional<CalibrationBounds> bounds = std::nullopt
);

}  // namespace vitavision
