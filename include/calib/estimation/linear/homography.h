/** @brief Linear homography estimation interfaces (DLT/RANSAC) */

#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/estimation/linear/planarpose.h"  // PlanarView
#include "calib/estimation/ransac.h"             // RansacOptions

namespace calib {

struct HomographyResult final {
    bool success{false};
    Eigen::Matrix3d hmtx = Eigen::Matrix3d::Identity();
    std::vector<int> inliers;      // indices of inlier correspondences
    double symmetric_rms_px{0.0};  // symmetric transfer RMS in pixels
};

auto estimate_homography(const PlanarView& data, std::optional<RansacOptions> ransac_opts = std::nullopt)
    -> HomographyResult;

}  // namespace calib

