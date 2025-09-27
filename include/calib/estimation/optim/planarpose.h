/** @brief Ceres-based planar pose optimization interfaces */

#pragma once

#include <Eigen/Geometry>

#include "calib/estimation/linear/planarpose.h"  // PlanarView and pose helpers
#include "calib/estimation/optimize.h"

namespace calib {

struct PlanarPoseOptions final : public OptimOptions {
    int num_radial = 2;  ///< Number of radial distortion coefficients
};

struct PlanarPoseResult final : public OptimResult {
    Eigen::Isometry3d pose;           ///< Estimated pose of the plane
    Eigen::VectorXd distortion;       ///< Estimated distortion coefficients
    double reprojection_error = 0.0;  ///< RMS reprojection error
};

auto optimize_planar_pose(const PlanarView& view, const CameraMatrix& intrinsics,
                          const Eigen::Isometry3d& init_pose, const PlanarPoseOptions& opts = {})
    -> PlanarPoseResult;

}  // namespace calib
