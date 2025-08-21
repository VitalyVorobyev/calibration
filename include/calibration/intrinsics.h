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

IntrinsicOptimizationResult optimize_intrinsics(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verb=false
);

}  // namespace vitavision
