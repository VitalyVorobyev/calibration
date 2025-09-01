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
auto pose_from_homography_normalized(const Eigen::Matrix3d& homography) -> Eigen::Affine3d;

// Convenience: one-shot planar pose from pixels & K
auto estimate_planar_pose_dlt(const std::vector<Eigen::Vector2d>& object_xy,
                             const std::vector<Eigen::Vector2d>& image_uv,
                             const CameraMatrix& intrinsics) -> Eigen::Affine3d;

// Convenience: one-shot planar pose from pixels & K
auto estimate_planar_pose_dlt(const PlanarView& observations, const CameraMatrix& intrinsics) -> Eigen::Affine3d;

struct PlanarPoseOptions final : public OptimOptions {
    int num_radial = 2;  ///< Number of radial distortion coefficients
};

struct PlanarPoseResult final : public OptimResult {
    Eigen::Affine3d pose;             ///< Estimated pose of the plane
    Eigen::VectorXd distortion;       ///< Estimated distortion coefficients
    double reprojection_error = 0.0;  ///< RMS reprojection error
};

auto optimize_planar_pose(const std::vector<Eigen::Vector2d>& object_xy,
                         const std::vector<Eigen::Vector2d>& image_uv,
                         const CameraMatrix& intrinsics,
                         const PlanarPoseOptions& opts = {}) -> PlanarPoseResult;

}  // namespace calib
