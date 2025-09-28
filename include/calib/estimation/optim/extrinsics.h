/** @brief Ceres-based multi-camera extrinsics optimization interfaces */

#pragma once

#include <Eigen/Geometry>
#include <vector>

#include "calib/estimation/linear/extrinsics.h"  // MulticamPlanarView
#include "calib/estimation/optim/optimize.h"
#include "calib/models/cameramodel.h"

namespace calib {

template <camera_model CameraT>
struct ExtrinsicOptimizationResult final : public OptimResult {
    std::vector<CameraT> cameras;            // Optimized camera matrices
    std::vector<Eigen::Isometry3d> c_se3_r;  // reference->camera
    std::vector<Eigen::Isometry3d> r_se3_t;  // target->reference
};

struct ExtrinsicOptions final : public OptimOptions {
    bool optimize_intrinsics = true;  ///< Solve for camera intrinsics
    bool optimize_skew = false;       ///< Solve for skew parameter
    bool optimize_extrinsics = true;  ///< Solve for camera and target extrinsics
};

template <camera_model CameraT>
auto optimize_extrinsics(const std::vector<MulticamPlanarView>& views,
                         const std::vector<CameraT>& init_cameras,
                         const std::vector<Eigen::Isometry3d>& init_c_se3_r,
                         const std::vector<Eigen::Isometry3d>& init_r_se3_t,
                         const ExtrinsicOptions& opts = {}) -> ExtrinsicOptimizationResult<CameraT>;

inline void to_json(nlohmann::json& j, const ExtrinsicOptions& o) {
    j = {{"optimize_intrinsics", o.optimize_intrinsics},
         {"optimize_skew", o.optimize_skew},
         {"optimize_extrinsics", o.optimize_extrinsics},
         {"huber_delta", o.huber_delta},
         {"epsilon", o.epsilon},
         {"max_iterations", o.max_iterations},
         {"compute_covariance", o.compute_covariance},
         {"verbose", o.verbose}};
}

inline void from_json(const nlohmann::json& j, ExtrinsicOptions& o) {
    o.optimize_intrinsics = j.value("optimize_intrinsics", o.optimize_intrinsics);
    o.optimize_skew = j.value("optimize_skew", o.optimize_skew);
    o.optimize_extrinsics = j.value("optimize_extrinsics", o.optimize_extrinsics);
    o.huber_delta = j.value("huber_delta", o.huber_delta);
    o.epsilon = j.value("epsilon", o.epsilon);
    o.max_iterations = j.value("max_iterations", o.max_iterations);
    o.compute_covariance = j.value("compute_covariance", o.compute_covariance);
    o.verbose = j.value("verbose", o.verbose);
}

template <camera_model CameraT>
inline void to_json(nlohmann::json& j, const ExtrinsicOptimizationResult<CameraT>& r) {
    j = {{"cameras", r.cameras},       {"c_se3_r", r.c_se3_r},       {"r_se3_t", r.r_se3_t},
         {"covariance", r.covariance}, {"final_cost", r.final_cost}, {"report", r.report}};
}

template <camera_model CameraT>
inline void from_json(const nlohmann::json& j, ExtrinsicOptimizationResult<CameraT>& r) {
    r.c_se3_r = j.value("c_se3_r", std::vector<Eigen::Isometry3d>{});
    r.r_se3_t = j.value("r_se3_t", std::vector<Eigen::Isometry3d>{});
    r.cameras = j.value("cameras", std::vector<CameraT>{});
    r.covariance = j.at("covariance").get<Eigen::MatrixXd>();
    r.final_cost = j.value("final_cost", 0.0);
    r.report = j.value("report", std::string{});
}

}  // namespace calib
