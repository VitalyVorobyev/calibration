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
using ExtrinsicPlanarView = std::vector<PlanarView>;

struct ExtrinsicPoses final {
    std::vector<Eigen::Affine3d> c_T_r;  // reference->camera
    std::vector<Eigen::Affine3d> r_T_t;  // target->reference
};

struct ExtrinsicOptimizationResult final : public OptimResult {
    ExtrinsicPoses poses;
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
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera<DualDistortion>>& cameras);

/**
 * @brief Optimizes the extrinsic poses of cameras given a set of planar views.
 *
 * This function refines the initial extrinsic poses of multiple cameras by minimizing
 * the reprojection error across a set of planar views. It uses the provided optimization
 * options to control the optimization process.
 *
 * @tparam CameraT Type representing the camera model.
 * @param views A vector of planar views containing observations for optimization.
 * @param cameras A vector of camera models corresponding to each view.
 * @param init_poses Initial estimates of the extrinsic poses for each camera.
 * @param options Options to configure the optimization process.
 * @return ExtrinsicOptimizationResult The result of the optimization, including refined poses and status.
 */
template<camera_model CameraT>
ExtrinsicOptimizationResult optimize_extrinsics(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<CameraT>& cameras,
    const ExtrinsicPoses& init_poses,
    const OptimOptions& options);

}  // namespace calib
