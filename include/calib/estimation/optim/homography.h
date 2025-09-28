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

inline void to_json(nlohmann::json& j, const HomographyOptions& o) {
    to_json(j, static_cast<const OptimOptions&>(o));
}

inline void from_json(const nlohmann::json& j, HomographyOptions& o) {
    from_json(j, static_cast<OptimOptions&>(o));
}

}  // namespace calib
