#pragma once

// std
#include <string>
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calib/pinhole.h"
#include "calib/planarpose.h"

namespace calib {

/** @brief Observation for line-scan calibration.
 * Contains planar target correspondences and laser line pixel features
 */
struct LineScanView final {
    PlanarView target_view;
    std::vector<Eigen::Vector2d> laser_uv;  // Pixel measurements of laser line points
};

struct LineScanCalibrationResult final {
    Eigen::Vector4d plane;       // Normalized plane coefficients [nx, ny, nz, d]
    Eigen::Matrix4d covariance;  // Covariance of plane coefficients
    Eigen::Matrix3d homography;  // Homography (norm pix -> plane frame)
    double rms_error;            // RMS distance of used points to fitted plane
    std::string summary;         // Optimizer report
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
auto calibrate_laser_plane(const std::vector<LineScanView>& views,
                           const PinholeCamera<DualDistortion>& camera)
    -> LineScanCalibrationResult;

}  // namespace calib
