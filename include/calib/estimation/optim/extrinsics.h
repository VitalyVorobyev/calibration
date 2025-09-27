/** @brief Ceres-based multi-camera extrinsics optimization interfaces */

#pragma once

#include <Eigen/Geometry>
#include <vector>

#include "calib/estimation/linear/extrinsics.h"  // MulticamPlanarView
#include "calib/estimation/optimize.h"
#include "calib/models/cameramodel.h"

namespace calib {

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

