#pragma once

// std
#include <vector>
#include <string>

// eigen
#include <Eigen/Geometry>

#include "calibration/planarpose.h"
#include "calibration/camera.h"

namespace vitavision {

struct ExtrinsicPlanarView final {
    // observations[camera][feature]
    std::vector<std::vector<PlanarObservation>> observations;
};

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

struct JointOptimizationResult final {
    std::vector<CameraMatrix> intrinsics;                       // Optimized camera matrices
    std::vector<Eigen::VectorXd> distortions;                   // Estimated distortion coeffs
    std::vector<Eigen::Affine3d> camera_poses;                  // reference->camera
    std::vector<Eigen::Affine3d> target_poses;                  // target->reference
    std::vector<Eigen::Matrix4d> intrinsic_covariances;         // Covariance of intrinsics
    std::vector<Eigen::Matrix<double,6,6>> camera_covariances;  // Covariance of camera poses
    std::vector<Eigen::Matrix<double,6,6>> target_covariances;  // Covariance of target poses
    double reprojection_error = 0.0;                            // RMS pixel error
    std::string summary;                                        // Solver brief report
};

InitialExtrinsicGuess make_initial_extrinsic_guess(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& cameras);

JointOptimizationResult optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& initial_cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose = false);

ExtrinsicOptimizationResult optimize_extrinsic_poses(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose = false);

}  // namespace vitavision
