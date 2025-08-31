#pragma once

#include "calib/extrinsics.h"

namespace calib {

struct JointOptimizationResult final {
    std::vector<CameraMatrix> intrinsics;                       // Optimized camera matrices
    std::vector<Eigen::VectorXd> distortions;                   // Estimated distortion coeffs
    std::vector<Eigen::Affine3d> camera_poses;                  // reference->camera
    std::vector<Eigen::Affine3d> target_poses;                  // target->reference
    std::vector<Eigen::Matrix<double,5,5>> intrinsic_covariances;         // Covariance of intrinsics
    std::vector<Eigen::Matrix<double,6,6>> camera_covariances;  // Covariance of camera poses
    std::vector<Eigen::Matrix<double,6,6>> target_covariances;  // Covariance of target poses
    double reprojection_error = 0.0;                            // RMS pixel error
    std::string summary;                                        // Solver brief report
};

JointOptimizationResult optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera<DualDistortion>>& initial_cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose = false);

}  // namespace calib
