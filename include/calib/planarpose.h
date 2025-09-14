/** @brief Estimate pose of a planar target using homography */

#pragma once

// eigen
#include <Eigen/Geometry>

#include "calib/cameramatrix.h"
#include "calib/optimize.h"

namespace calib {

struct PlanarObservation {
    Eigen::Vector2d object_xy;  // Planar target coordinates (Z=0)
    Eigen::Vector2d image_uv;   // Corresponding pixel measurements
};
using PlanarView = std::vector<PlanarObservation>;

// Decompose homography in normalized camera coords: H = [r1 r2 t]
auto pose_from_homography_normalized(const Eigen::Matrix3d& hmtx) -> Eigen::Isometry3d;

// Convenience: one-shot planar pose from pixels & kmtx
auto estimate_planar_pose(PlanarView view, const CameraMatrix& intrinsics) -> Eigen::Isometry3d;

struct PlanarPoseOptions final : public OptimOptions {
    int num_radial = 2;  ///< Number of radial distortion coefficients
};

struct PlanarPoseResult final : public OptimResult {
    Eigen::Isometry3d pose;           ///< Estimated pose of the plane
    Eigen::VectorXd distortion;       ///< Estimated distortion coefficients
    double reprojection_error = 0.0;  ///< RMS reprojection error
};

auto optimize_planar_pose(const PlanarView& view, const CameraMatrix& intrinsics,
                          const Eigen::Isometry3d& init_pose,
                          const PlanarPoseOptions& opts = {}) -> PlanarPoseResult;

}  // namespace calib
