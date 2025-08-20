/** @brief Optimization of camera intrinsics parameters */

#pragma once

// std
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calibration/distortion.h"  // Observation

namespace vitavision {

struct Intrinsic {
    double fx, fy, cx, cy;

    Eigen::Vector2d pixel_to_norm(const Eigen::Vector2d& pix) const {
        return Eigen::Vector2d((pix.x() - cx) / fx, (pix.y() - cy) / fy);
    }
};

struct IntrinsicOptimizationResult {
    Intrinsic intrinsics;
    Eigen::VectorXd distortion;
    Eigen::Matrix4d covariance;  // Covariance matrix of intrinsics
    double reprojection_error;   // Reprojection error after optimization (pix)
    std::string summary;         // Summary of optimization results
};

IntrinsicOptimizationResult optimize_intrinsics(
    const std::vector<Observation>& obs,
    int num_radial,
    const Intrinsic& initial_guess,
    bool verb=false
);

}  // namespace vitavision
