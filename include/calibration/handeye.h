#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calibration/calib.h" // for PlanarView
#include "calibration/intrinsics.h"

namespace vitavision {

/**
 * \brief Single observation used for hand-eye calibration.
 *
 * Each observation corresponds to a planar target view acquired by one of the
 * cameras when the robot end-effector is in a known pose relative to the base
 * frame.
 */
struct HandEyeObservation final {
    PlanarView view;                ///< Planar target observations
    Eigen::Affine3d base_T_gripper; ///< Pose of the gripper in the base frame
    size_t camera_index = 0;        ///< Which camera acquired this view
};

/** Options controlling the hand-eye calibration optimisation. */
struct HandEyeOptions final {
    bool optimize_intrinsics = false;    ///< Solve for camera intrinsics
    bool optimize_target_pose = true;    ///< Solve for base->target pose
    bool optimize_hand_eye = true;       ///< Solve for gripper->camera pose
    bool optimize_extrinsics = true;     ///< Solve for reference->camera extrinsics
    bool verbose = false;                ///< Verbose solver output
};

/** Result returned by hand-eye calibration. */
struct HandEyeResult final {
    std::vector<CameraMatrix> intrinsics;          ///< Estimated intrinsics per camera
    std::vector<Eigen::VectorXd> distortions;      ///< Estimated distortion coefficients
    std::vector<Eigen::Affine3d> hand_eye;         ///< Estimated gripper->camera transforms
    std::vector<Eigen::Affine3d> extrinsics;       ///< Estimated reference->camera extrinsics
    Eigen::Affine3d base_T_target = Eigen::Affine3d::Identity(); ///< Pose of target in base frame
    double reprojection_error = 0.0;               ///< RMSE of reprojection
    std::string summary;                           ///< Ceres summary
    Eigen::MatrixXd covariance;                    ///< Covariance of pose parameters
};

/**
 * Compute an initial estimate of the hand-eye transform (gripper -> camera)
 * using the Tsai-Lenz linear method.  The input vectors must contain
 * corresponding poses of the robot end-effector in the base frame and poses of
 * the planar target in the camera frame for the same time instants.
 */
Eigen::Affine3d estimate_hand_eye_initial(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& target_T_camera);

/**
 * Perform bundle-adjustment style optimisation of the hand-eye calibration
 * problem.  Supports single or multiple cameras and optional optimisation of
 * intrinsics and the target pose.
 */
HandEyeResult calibrate_hand_eye(
    const std::vector<HandEyeObservation>& observations,
    const std::vector<CameraMatrix>& initial_intrinsics,
    const Eigen::Affine3d& initial_hand_eye,
    const std::vector<Eigen::Affine3d>& initial_extrinsics = {},
    const Eigen::Affine3d& initial_base_target = Eigen::Affine3d::Identity(),
    const HandEyeOptions& opts = {});

} // namespace vitavision
