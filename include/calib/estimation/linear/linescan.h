/**
 * @file linescan.h
 * @brief Linear laser plane calibration from line-scan observations
 *
 * Provides simple, dependency-free routines to calibrate a laser plane
 * observed by a camera when a planar target is also visible. This module
 * performs only linear or closed-form computations (homography via DLT,
 * plane via SVD), making it suitable for fast initialization and tests.
 */

#pragma once

// std
#include <string>
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calib/estimation/planarpose.h"     // PlanarView/Observation
#include "calib/estimation/homography.h"     // estimate_homography
#include "calib/models/pinhole.h"            // PinholeCamera

namespace calib {

/**
 * @brief Observation for laser plane calibration.
 *
 * A single view consists of planar target correspondences (object XY on the
 * target plane and image UV in pixels) and pixel coordinates of detected
 * laser points in the same image.
 */
struct LineScanView final {
    PlanarView target_view;               ///< Target correspondences
    std::vector<Eigen::Vector2d> laser_uv;  ///< Laser points in pixels
};

/**
 * @brief Result of linear laser plane calibration.
 */
struct LineScanCalibrationResult final {
    Eigen::Vector4d plane;       ///< Normalized plane [nx, ny, nz, d], ||n||=1
    Eigen::Matrix4d covariance;  ///< Approximate covariance (zero if unavailable)
    Eigen::Matrix3d homography;  ///< Map from normalized pixels to plane frame
    double rms_error = 0.0;      ///< RMS point-to-plane distance of fitted points
    std::string summary;         ///< Text report
};

/**
 * @brief Calibrate a laser plane in camera frame from multiple views (linear).
 *
 * Steps per view:
 * 1) Undistort/normalize target pixel coordinates using camera model.
 * 2) Estimate homography (DLT) from normalized pixels to target plane.
 * 3) Reproject laser pixels to 3D points on the target plane and transform to camera frame.
 *
 * After aggregating points from all views, a plane is fitted by SVD. The
 * reported covariance is set to zero (for a robust covariance, use an
 * optimization-based refinement outside of this linear module).
 *
 * @param views   Vector of line-scan views
 * @param camera  Camera model (pinhole with dual distortion)
 * @return Calibration result with plane, RMS, homography
 * @throws std::invalid_argument if too few views or insufficient correspondences
 */
auto calibrate_laser_plane(const std::vector<LineScanView>& views,
                           const PinholeCamera<DualDistortion>& camera)
    -> LineScanCalibrationResult;

// JSON helpers are intentionally not provided here to keep this header
// dependency-free. Examples may implement their own parsing.

}  // namespace calib
