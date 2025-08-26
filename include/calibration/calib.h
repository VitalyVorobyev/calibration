#pragma once

/** @brief Full single camera calibration from multiple planar views */

// std
#include <vector>
#include <string>
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calibration/intrinsics.h"
#include "calibration/planarpose.h"  // PlanarObservation

namespace vitavision {

struct CameraCalibrationResult final {
    CameraMatrix intrinsics;             // Estimated camera matrix
    Eigen::VectorXd distortion;          // Distortion coefficients [k..., p1, p2]
    std::vector<Eigen::Affine3d> poses;  // Poses of each view (world->camera)
    Eigen::MatrixXd covariance;          // Covariance of intrinsics and poses
    std::vector<double> view_errors;     // Reprojection RMSE for each view
    double reprojection_error;           // Overall reprojection RMSE
    std::string summary;                 // Solver brief report
};

CameraCalibrationResult calibrate_camera_planar(
    const std::vector<PlanarView>& views,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verbose = false,
    std::optional<CalibrationBounds> bounds = std::nullopt);

}  // namespace vitavision
