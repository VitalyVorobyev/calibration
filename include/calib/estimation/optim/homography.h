/** @brief Ceres-based homography optimization interfaces */

#pragma once

#include <Eigen/Core>

#include "calib/estimation/linear/homography.h"  // PlanarView
#include "calib/estimation/optim/optimize.h"

namespace calib {

struct HomographyOptions final : public OptimOptions {};

struct OptimizeHomographyResult final : OptimResult {
    Eigen::Matrix3d homography;
};

auto optimize_homography(const PlanarView& data, const Eigen::Matrix3d& init_h,
                         const HomographyOptions& options = {}) -> OptimizeHomographyResult;

}  // namespace calib
