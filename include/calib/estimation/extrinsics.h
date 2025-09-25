#pragma once

// std
#include <string>
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/estimation/planarpose.h"
#include "calib/estimation/optimize.h"
#include "calib/models/pinhole.h"

namespace calib {

// [camera]
using MulticamPlanarView = std::vector<PlanarView>;

struct ExtrinsicPoses final {
    std::vector<Eigen::Isometry3d> c_se3_r;  // reference->camera
    std::vector<Eigen::Isometry3d> r_se3_t;  // target->reference
};

/**
 * @brief Estimates the extrinsic poses of cameras using the Direct Linear Transform (DLT)
 * algorithm.
 *
 * This function computes the extrinsic parameters (rotation and translation) for a set of cameras
 * given multiple planar views and their corresponding camera models with dual distortion.
 *
 * @param views A vector of planar views containing the observed 2D-3D correspondences for each
 * camera.
 * @param cameras A vector of camera models, each with dual distortion parameters, corresponding to
 * the views.
 * @return ExtrinsicPoses The estimated extrinsic poses (rotation and translation) for each camera.
 */
auto estimate_extrinsic_dlt(const std::vector<MulticamPlanarView>& views,
                            const std::vector<Camera<DualDistortion>>& cameras) -> ExtrinsicPoses;

template <camera_model CameraT>
struct ExtrinsicOptimizationResult final : public OptimResult {
    std::vector<CameraT> cameras;            // Optimized camera matrices
    std::vector<Eigen::Isometry3d> c_se3_r;  // reference->camera
    std::vector<Eigen::Isometry3d> r_se3_t;  // target->reference
};

struct ExtrinsicOptions final : public OptimOptions {
    bool optimize_intrinsics = true;  ///< Solve for camera intrinsics
    bool optimize_skew = false;       ///< Solve for skew parameter
    bool optimize_extrinsics = true;  ///< Solve for camera and target extrinsics
};

template <camera_model CameraT>
auto optimize_extrinsics(const std::vector<MulticamPlanarView>& views,
                         const std::vector<CameraT>& init_cameras,
                         const std::vector<Eigen::Isometry3d>& init_c_se3_r,
                         const std::vector<Eigen::Isometry3d>& init_r_se3_t,
                         const ExtrinsicOptions& opts = {}) -> ExtrinsicOptimizationResult<CameraT>;

}  // namespace calib
