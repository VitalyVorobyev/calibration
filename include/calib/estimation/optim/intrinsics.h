/** @brief Ceres-based intrinsic optimization interfaces */

#pragma once

#include <vector>

#include "calib/estimation/linear/intrinsics.h"
#include "calib/estimation/optim/optimize.h"

namespace calib {

struct IntrinsicsOptions final : public OptimOptions {
    int num_radial = 2;          ///< Number of radial distortion coefficients
    bool optimize_skew = false;  ///< Estimate skew parameter
    std::optional<CalibrationBounds> bounds = std::nullopt;  ///< Parameter bounds
    std::vector<int> fixed_distortion_indices;               ///< Indices of coeffs to keep fixed
    std::vector<double> fixed_distortion_values;             ///< Assigned fixed values
};

template <camera_model CameraT>
struct IntrinsicsOptimizationResult final : public OptimResult {
    CameraT camera;                          ///< Estimated camera parameters
    std::vector<Eigen::Isometry3d> c_se3_t;  ///< Estimated pose of each view
    std::vector<double> view_errors;         ///< Per-view reprojection errors
};

auto optimize_intrinsics_semidlt(const std::vector<PlanarView>& views,
                                 const CameraMatrix& initial_guess,
                                 const IntrinsicsOptions& opts = {})
    -> IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>>;

template <camera_model CameraT>
auto optimize_intrinsics(const std::vector<PlanarView>& views, const CameraT& init_camera,
                         std::vector<Eigen::Isometry3d> init_c_se3_t,
                         const IntrinsicsOptions& opts = {})
    -> IntrinsicsOptimizationResult<CameraT>;

}  // namespace calib
