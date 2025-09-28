#pragma once

#include <Eigen/Core>
#include <limits>
#include <vector>

#include "calib/estimation/common/ransac.h"
#include "calib/models/cameramodel.h"

namespace calib {

auto fit_plane_svd(const std::vector<Eigen::Vector3d>& pts) -> Eigen::Vector4d;

struct PlaneRansacResult final {
    bool success{false};
    Eigen::Vector4d plane{Eigen::Vector4d::Zero()};
    std::vector<int> inliers;
    double inlier_rms{std::numeric_limits<double>::infinity()};
};

auto fit_plane_ransac(const std::vector<Eigen::Vector3d>& pts, const RansacOptions& opts = {})
    -> PlaneRansacResult;

}  // namespace calib
