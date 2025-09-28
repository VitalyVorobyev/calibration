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


template <camera_model CameraT>
inline void to_json(nlohmann::json& j, const IntrinsicsOptimizationResult<CameraT>& r) {
    j = {{"camera", r.camera},           {"poses", r.c_se3_t},         {"covariance", r.covariance},
         {"view_errors", r.view_errors}, {"final_cost", r.final_cost}, {"report", r.report}};
}

template <camera_model CameraT>
inline void from_json(const nlohmann::json& j, IntrinsicsOptimizationResult<CameraT>& r) {
    j.at("camera").get_to(r.camera);
    r.c_se3_t = j.value("poses", std::vector<Eigen::Isometry3d>{});
    r.covariance = j.at("covariance").get<Eigen::MatrixXd>();
    r.view_errors = j.value("view_errors", std::vector<double>{});
    r.final_cost = j.value("final_cost", 0.0);
    r.report = j.value("report", std::string{});
}

}  // namespace calib
