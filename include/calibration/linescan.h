#pragma once

// std
#include <vector>
#include <string>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calibration/intrinsics.h"

namespace vitavision {

/** @brief Observation for line-scan calibration.
 * Contains planar target correspondences and laser line pixel features
 */
struct LineScanObservation {
    std::vector<Eigen::Vector2d> target_xy;  // Target plane coordinates (Z=0)
    std::vector<Eigen::Vector2d> target_uv;  // Pixel measurements of target points
    std::vector<Eigen::Vector2d> laser_uv;   // Pixel measurements of laser line points
};

struct LineScanCalibrationResult {
    Eigen::Vector4d plane;                  // Normalized plane coefficients [nx, ny, nz, d]
    Eigen::Matrix4d covariance;             // Covariance of plane coefficients
    Eigen::Matrix3d homography;             // Homography (norm pix -> plane frame)
    double rms_error;                       // RMS distance of used points to fitted plane
    std::string summary;                    // Optimizer report
};

/**
 * @brief Calibrate a laser plane in camera frame using planar target views.
 *
 * For each observation, target feature correspondences are used to convert
 * laser pixels into 3D points in the camera frame. All points from all
 * observations are used to fit a single plane using non-linear least squares
 * with automatic differentiation. A local 2D reference frame is introduced on
 * the fitted plane and the homography mapping undistorted pixel coordinates to
 * this frame is returned. The plane coefficients are normalised so that
 * ||n|| = 1.
 */
LineScanCalibrationResult calibrate_laser_plane(
    const std::vector<LineScanObservation>& views,
    const CameraMatrix& intrinsics);

} // namespace vitavision

