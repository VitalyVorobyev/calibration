#pragma once

#include "calib/extrinsics.h"

#include "calib/optimize.h"

namespace calib {

template<camera_model CameraT>
struct JointOptimizationResult final : public OptimResult {
    std::vector<CameraT> cameras;       // Optimized camera matrices
    std::vector<Eigen::Affine3d> c_T_r;  // reference->camera
    std::vector<Eigen::Affine3d> r_T_t;  // target->reference
    #if 0
    std::vector<Eigen::Matrix<double,5,5>> intrinsic_covariances; // Covariance of intrinsics
    std::vector<Eigen::Matrix<double,6,6>> camera_covariances;  // Covariance of camera poses
    std::vector<Eigen::Matrix<double,6,6>> target_covariances;  // Covariance of target poses
    #endif
};

struct JointIntrExtrOptions final : public OptimOptions {
    bool optimize_intrinsics = true;  ///< Solve for camera intrinsics
    bool optimize_skew = false;       ///< Solve for skew parameter
    bool optimize_extrinsics = true;  ///< Solve for camera and target extrinsics
};

template<camera_model CameraT>
JointOptimizationResult<CameraT> optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<CameraT>& initial_cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    const JointIntrExtrOptions& opts = {});

}  // namespace calib
