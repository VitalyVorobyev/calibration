/** @brief Optimization of camera intrinsics parameters */

#pragma once

// std
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calibration/distortion.h"  // Observation

namespace vitavision {

struct CameraMatrix {
    double fx, fy, cx, cy;

    Eigen::Vector2d normalize(const Eigen::Vector2d& pix) const;
    Eigen::Vector2d denormalize(const Eigen::Vector2d& xy) const;
};

struct IntrinsicOptimizationResult {
    CameraMatrix intrinsics;
    Eigen::VectorXd distortion;
    Eigen::Matrix4d covariance;  // Covariance matrix of intrinsics
    double reprojection_error;   // Reprojection error after optimization (pix)
    std::string summary;         // Summary of optimization results
};

// Estimate camera intrinsics (fx, fy, cx, cy) by solving a linear
// least-squares system that ignores lens distortion.  The input
// observations contain normalized coordinates (x,y) for an undistorted
// point and the observed pixel coordinates (u,v).  The function returns
// an optional CameraMatrix: std::nullopt is returned if there are not
// enough observations or the linear system is degenerate.
std::optional<CameraMatrix> estimate_intrinsics_linear(
    const std::vector<Observation<double>>& obs);

IntrinsicOptimizationResult optimize_intrinsics(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verb=false
);

}  // namespace vitavision
