#pragma once

// std
#include <vector>
#include <string>

// eigen
#include <Eigen/Geometry>

#include "calibration/planarpose.h"
#include "calibration/camera.h"

namespace vitavision {

// [camera]
using ExtrinsicPlanarView = std::vector<PlanarView>;

struct ExtrinsicOptimizationResult final {
    std::vector<Eigen::Affine3d> camera_poses;   // reference->camera
    std::vector<Eigen::Affine3d> target_poses;   // target->reference
    std::vector<Eigen::Matrix<double,6,6>> camera_covariances; // Covariance of camera poses
    std::vector<Eigen::Matrix<double,6,6>> target_covariances; // Covariance of target poses
    double reprojection_error = 0.0;             // RMS pixel error
    std::string summary;                         // Solver brief report
};

struct InitialExtrinsicGuess final {
    std::vector<Eigen::Affine3d> camera_poses;  // reference->camera
    std::vector<Eigen::Affine3d> target_poses;  // target->reference
};

InitialExtrinsicGuess make_initial_extrinsic_guess(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& cameras);

ExtrinsicOptimizationResult optimize_extrinsic_poses(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose = false);

}  // namespace vitavision
