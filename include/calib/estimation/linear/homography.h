/** @brief Linear homography estimation interfaces (DLT/RANSAC) */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <optional>
#include <vector>

#include "calib/estimation/common/ransac.h"      // RansacOptions
#include "calib/estimation/linear/planarpose.h"  // PlanarView

namespace calib {

struct HomographyResult final {
    bool success{false};
    Eigen::Matrix3d hmtx = Eigen::Matrix3d::Identity();
    std::vector<int> inliers;      // indices of inlier correspondences
    double symmetric_rms_px{0.0};  // symmetric transfer RMS in pixels
};

auto estimate_homography(const PlanarView& data,
                         std::optional<RansacOptions> ransac_opts = std::nullopt)
    -> HomographyResult;

}  // namespace calib
