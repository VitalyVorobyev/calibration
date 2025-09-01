#pragma once

// std
#include <vector>
#include <string>

// eigen
#include <Eigen/Geometry>

#include "calib/planarpose.h"
#include "calib/camera.h"

#include "calib/optimize.h"

namespace calib {

// [camera]
using MulticamPlanarView = std::vector<PlanarView>;

struct ExtrinsicPoses final {
    std::vector<Eigen::Affine3d> c_T_r;  // reference->camera
    std::vector<Eigen::Affine3d> r_T_t;  // target->reference
};

/**
 * @brief Estimates the extrinsic poses of cameras using the Direct Linear Transform (DLT) algorithm.
 *
 * This function computes the extrinsic parameters (rotation and translation) for a set of cameras
 * given multiple planar views and their corresponding camera models with dual distortion.
 *
 * @param views A vector of planar views containing the observed 2D-3D correspondences for each camera.
 * @param cameras A vector of camera models, each with dual distortion parameters, corresponding to the views.
 * @return ExtrinsicPoses The estimated extrinsic poses (rotation and translation) for each camera.
 */
ExtrinsicPoses estimate_extrinsic_dlt(
    const std::vector<MulticamPlanarView>& views,
    const std::vector<Camera<DualDistortion>>& cameras);

template<camera_model CameraT>
struct ExtrinsicOptimizationResult final : public OptimResult {
    std::vector<CameraT> cameras;       // Optimized camera matrices
    std::vector<Eigen::Affine3d> c_T_r;  // reference->camera
    std::vector<Eigen::Affine3d> r_T_t;  // target->reference
};

struct ExtrinsicOptions final : public OptimOptions {
    bool optimize_intrinsics = true;  ///< Solve for camera intrinsics
    bool optimize_skew = false;       ///< Solve for skew parameter
    bool optimize_extrinsics = true;  ///< Solve for camera and target extrinsics
};

template<camera_model CameraT>
ExtrinsicOptimizationResult<CameraT> optimize_extrinsics(
    const std::vector<MulticamPlanarView>& views,
    const std::vector<CameraT>& initial_cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    const ExtrinsicOptions& opts = {});

}  // namespace calib
