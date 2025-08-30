#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calibration/camera.h"
#include "calibration/planarpose.h"  // PlanarObservation

namespace vitavision {

/**
 * \brief Single observation used for hand-eye calibration.
 *
 * Each observation corresponds to a planar target view acquired by one of the
 * cameras when the robot end-effector is in a known pose relative to the base
 * frame.
 */
struct BundleObservation final {
    PlanarView view;          ///< Planar target observations
    Eigen::Affine3d b_T_g;    ///< Pose of the gripper in the base frame
    size_t camera_index = 0;  ///< Which camera acquired this view
};

/** Options controlling the hand-eye calibration optimisation. */
struct BundleOptions final {
    bool optimize_intrinsics = false;  ///< Solve for camera intrinsics
    bool optimize_target_pose = true;  ///< Solve for base->target pose
    bool optimize_hand_eye = true;     ///< Solve for gripper->camera pose
    bool optimize_extrinsics = true;   ///< Solve for reference->camera extrinsics
    bool verbose = false;              ///< Verbose solver output
};

/** Result returned by hand-eye calibration. */
struct BundleResult final {
    std::vector<Camera> cameras;               ///< Estimated camera parameters per camera
    Eigen::Affine3d g_T_r;                     ///< Estimated reference camera -> gripper transforms
    std::vector<Eigen::Affine3d> c_T_r;        ///< Estimated reference->camera extrinsics
    Eigen::Affine3d b_T_t;                     ///< Pose of target in base frame
    double reprojection_error = 0.0;           ///< RMSE of reprojection
    Eigen::MatrixXd covariance;                ///< Covariance of pose parameters
    std::string report;                        ///< Ceres summary
};

/**
 * Perform bundle-adjustment style optimisation of the hand-eye calibration
 * problem.  Supports single or multiple cameras and optional optimisation of
 * intrinsics and the target pose.
 * @param observations Set of observations with robot poses and target detections
 * @param initial_cameras Initial camera parameters
 * @param init_g_T_r Initial estimate of hand-eye transformation
 * @param init_c_T_r Initial estimates of reference camera to camera transformations
 * @param init_b_T_t Initial estimate of base-to-target transformation
 * @param opts Optimization options
 * @return Calibration result containing optimized parameters and error metrics
 */
BundleResult optimize_bundle(
    const std::vector<BundleObservation>& observations,
    const std::vector<Camera>& initial_cameras,
    const Eigen::Affine3d& init_g_T_r,
    const std::vector<Eigen::Affine3d>& init_c_T_r = {},
    const Eigen::Affine3d& init_b_T_t = Eigen::Affine3d::Identity(),
    const BundleOptions& opts = {});

} // namespace vitavision
