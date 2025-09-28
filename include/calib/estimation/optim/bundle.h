#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/estimation/linear/planarpose.h"  // PlanarObservation
#include "calib/estimation/optim/optimize.h"
#include "calib/models/cameramodel.h"
#include "calib/models/pinhole.h"
#include "calib/models/scheimpflug.h"

namespace calib {

/**
 * \brief Single observation used for hand-eye calibration.
 *
 * Each observation corresponds to a planar target view acquired by one of the
 * cameras when the robot end-effector is in a known pose relative to the base
 * frame.
 */
struct BundleObservation final {
    PlanarView view;            ///< Planar target observations
    Eigen::Isometry3d b_se3_g;  ///< Pose of the gripper in the base frame
    size_t camera_index = 0;    ///< Which camera acquired this view
};

/** Options controlling the hand-eye calibration optimisation. */
// TODO: add optimize_distortion
struct BundleOptions final : public OptimOptions {
    bool optimize_intrinsics = false;  ///< Solve for camera intrinsics
    bool optimize_skew = false;        ///< Solve for skew parameter
    bool optimize_target_pose = true;  ///< Solve for base->target pose
    bool optimize_hand_eye = true;     ///< Solve for camera->gripper poses
};

/** Result returned by hand-eye calibration. */
template <camera_model CameraT>
struct BundleResult final : public OptimResult {
    std::vector<CameraT> cameras;            ///< Estimated camera parameters per camera
    std::vector<Eigen::Isometry3d> g_se3_c;  ///< Estimated camera->gripper extrinsics
    Eigen::Isometry3d b_se3_t;               ///< Pose of target in base frame
};

/**
 * Perform bundle-adjustment style optimisation of the hand-eye calibration
 * problem.  Supports single or multiple cameras and optional optimisation of
 * intrinsics and the target pose.
 * @param observations Set of observations with robot poses and target detections
 * @param initial_cameras Initial camera parameters
 * @param init_g_se3_c Initial estimate of hand-eye transformations
 * @param init_b_se3_t Initial estimate of base-to-target transformation
 * @param opts Optimization options
 * @return Calibration result containing optimized parameters and error metrics
 */
template <camera_model CameraT>
auto optimize_bundle(const std::vector<BundleObservation>& observations,
                     const std::vector<CameraT>& initial_cameras,
                     const std::vector<Eigen::Isometry3d>& init_g_se3_c,
                     const Eigen::Isometry3d& init_b_se3_t, const BundleOptions& opts = {})
    -> BundleResult<CameraT>;


inline void to_json(nlohmann::json& j, const BundleOptions& o) {
    j = {{"optimize_intrinsics", o.optimize_intrinsics},
         {"optimize_skew", o.optimize_skew},
         {"optimize_target_pose", o.optimize_target_pose},
         {"optimize_hand_eye", o.optimize_hand_eye},
         {"verbose", o.verbose}};
}

inline void from_json(const nlohmann::json& j, BundleOptions& o) {
    o.optimize_intrinsics = j.value("optimize_intrinsics", false);
    o.optimize_skew = j.value("optimize_skew", false);
    o.optimize_target_pose = j.value("optimize_target_pose", true);
    o.optimize_hand_eye = j.value("optimize_hand_eye", true);
    o.verbose = j.value("verbose", false);
}

inline void to_json(nlohmann::json& j, const BundleObservation& bo) {
    j = {{"view", bo.view}, {"b_se3_g", bo.b_se3_g}, {"camera_index", bo.camera_index}};
}

inline void from_json(const nlohmann::json& j, BundleObservation& bo) {
    j.at("view").get_to(bo.view);
    bo.b_se3_g = j.at("b_se3_g").get<Eigen::Isometry3d>();
    bo.camera_index = j.value("camera_index", 0);
}

inline void to_json(nlohmann::json& j, const BundleResult<PinholeCamera<BrownConradyd>>& r) {
    j = {{"cameras", r.cameras},       {"g_se3_c", r.g_se3_c},       {"b_se3_t", r.b_se3_t},
         {"final_cost", r.final_cost}, {"covariance", r.covariance}, {"report", r.report}};
}

inline void from_json(const nlohmann::json& j, BundleResult<PinholeCamera<BrownConradyd>>& r) {
    r.cameras = j.value("cameras", std::vector<PinholeCamera<BrownConradyd>>{});
    r.g_se3_c = j.value("g_se3_c", std::vector<Eigen::Isometry3d>{});
    r.b_se3_t = j.at("b_se3_t").get<Eigen::Isometry3d>();
    r.final_cost = j.value("final_cost", 0.0);
    r.covariance = j.at("covariance").get<Eigen::MatrixXd>();
    r.report = j.value("report", std::string{});
}

}  // namespace calib
