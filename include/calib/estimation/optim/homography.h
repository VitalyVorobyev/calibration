/** @brief Ceres-based homography optimization interfaces */

#pragma once

#include <Eigen/Core>

#include "calib/estimation/linear/homography.h"  // PlanarView
#include "calib/estimation/optim/optimize.h"

namespace calib {

struct OptimizeHomographyResult final {
    OptimResult core;  ///< Core optimization result
    Eigen::Matrix3d homography;
};

auto optimize_homography(const PlanarView& data, const Eigen::Matrix3d& init_h,
                         const OptimOptions& options = {}) -> OptimizeHomographyResult;

}  // namespace calib
