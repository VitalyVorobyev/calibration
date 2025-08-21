/** @brief Estimate pose of a planar target using homography */

#pragma once

// eigen
#include <Eigen/Geometry>

#include "calibration/intrinsics.h"

namespace vitavision {

struct PlanarObservation {
    Eigen::Vector2d object_xy;  // Planar target coordinates (Z=0)
    Eigen::Vector2d image_uv;   // Corresponding pixel measurements
};

// Decompose homography in normalized camera coords: H = [r1 r2 t]
Eigen::Affine3d pose_from_homography_normalized(const Eigen::Matrix3d& H);

// Convenience: one-shot planar pose from pixels & K
Eigen::Affine3d estimate_planar_pose_dlt(const std::vector<Eigen::Vector2d>& obj_xy,
                                         const std::vector<Eigen::Vector2d>& img_uv,
                                         const CameraMatrix& intrinsics);

struct PlanarPoseFitResult {
    Eigen::Affine3d pose;
    Eigen::VectorXd distortion;
    Eigen::Matrix<double, 6, 6> covariance;  // Covariance matrix of axis-and-angle and translation
    double reprojection_error;
    std::string summary;  // Summary of optimization results
};

PlanarPoseFitResult optimize_planar_pose(
    const std::vector<Eigen::Vector2d>& obj_xy,
    const std::vector<Eigen::Vector2d>& img_uv,
    const CameraMatrix& intrinsics,
    int num_radial = 2,
    bool verbose = false
);

}  // namespace vitavision
