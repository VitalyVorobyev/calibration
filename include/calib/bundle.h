#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/cameramodel.h"
#include "calib/optimize.h"
#include "calib/pinhole.h"
#include "calib/planarpose.h"  // PlanarObservation
#include "calib/scheimpflug.h"

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

}  // namespace calib
