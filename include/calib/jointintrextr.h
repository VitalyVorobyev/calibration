#pragma once

#include "calib/extrinsics.h"
#include "calib/bundle.h"  // OptimizerType

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

struct JointOptions final {
    double epsilon = 1e-9;                 ///< Solver convergence tolerance
    int max_iterations = 1000;             ///< Maximum number of iterations
    OptimizerType optimizer = OptimizerType::DENSE_QR; ///< Linear solver type
    bool verbose = false;                  ///< Verbose solver output
};

JointOptimizationResult optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera<DualDistortion>>& initial_cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    const JointOptions& opts = {});

}  // namespace calib
