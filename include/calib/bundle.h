#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/camera.h"
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
    PlanarView view;          ///< Planar target observations
    Eigen::Affine3d b_T_g;    ///< Pose of the gripper in the base frame
    size_t camera_index = 0;  ///< Which camera acquired this view
};

enum class OptimizerType {
    DEFAULT,  // SPARSE_NORMAL_CHOLESKY
    SPARSE_SCHUR,  // for large problems
    DENSE_SCHUR,  // for small multiple camera problems
    DENSE_QR  // for small single camera problems
};

/** Options controlling the hand-eye calibration optimisation. */
struct BundleOptions final {
    bool optimize_intrinsics = false;  ///< Solve for camera intrinsics
    bool optimize_target_pose = true;  ///< Solve for base->target pose
    bool optimize_hand_eye = true;     ///< Solve for camera->gripper poses
    double huber_delta = 1.0;          ///< Huber loss delta
    double epsilon = 1e-9;             ///< Solver convergence tolerance
    int max_iterations = 1000;         ///< Maximum number of iterations
    OptimizerType optimizer = OptimizerType::DEFAULT;
    bool verbose = false;              ///< Verbose solver output
};

/** Result returned by hand-eye calibration. */
struct BundleResult final {
    std::vector<Camera<BrownConradyd>> cameras;               ///< Estimated camera parameters per camera
    std::vector<Eigen::Affine3d> g_T_c;        ///< Estimated camera->gripper extrinsics
    Eigen::Affine3d b_T_t;                     ///< Pose of target in base frame
    double reprojection_error = 0.0;           ///< RMSE of reprojection
    Eigen::MatrixXd covariance;                ///< Covariance of pose parameters
    std::string report;                        ///< Ceres summary
};

struct ScheimpflugBundleResult final {
    std::vector<ScheimpflugCamera<BrownConradyd>> cameras;   ///< Estimated cameras with tilt
    std::vector<Eigen::Affine3d> g_T_c;       ///< Estimated camera->gripper extrinsics
    Eigen::Affine3d b_T_t;                    ///< Pose of target in base frame
    double reprojection_error = 0.0;          ///< RMSE of reprojection
    Eigen::MatrixXd covariance;               ///< Covariance of pose parameters
    std::string report;                       ///< Ceres summary
};

/**
 * Perform bundle-adjustment style optimisation of the hand-eye calibration
 * problem.  Supports single or multiple cameras and optional optimisation of
 * intrinsics and the target pose.
 * @param observations Set of observations with robot poses and target detections
 * @param initial_cameras Initial camera parameters
 * @param init_g_T_c Initial estimate of hand-eye transformations
 * @param init_b_T_t Initial estimate of base-to-target transformation
 * @param opts Optimization options
 * @return Calibration result containing optimized parameters and error metrics
 */
BundleResult optimize_bundle(
    const std::vector<BundleObservation>& observations,
    const std::vector<Camera<BrownConradyd>>& initial_cameras,
    const std::vector<Eigen::Affine3d>& init_g_T_c,
    const Eigen::Affine3d& init_b_T_t,
    const BundleOptions& opts = {});

/**
 * @brief Optimizes the bundle adjustment for a Scheimpflug camera setup.
 *
 * This function performs bundle adjustment optimization for a set of observations
 * captured by Scheimpflug cameras. It refines the camera parameters, poses, and
 * other related transformations to minimize the reprojection error.
 *
 * @param observations A vector of BundleObservation objects representing the
 *                     observed 2D points and their corresponding 3D points.
 * @param initial_cameras A vector of ScheimpflugCamera<BrownConradyd> objects
 *                        representing the initial intrinsic parameters of the cameras.
 * @param init_g_T_c A vector of Eigen::Affine3d transformations representing the
 *                   initial guesses for the camera poses in the global frame.
 * @param init_b_T_t An Eigen::Affine3d transformation representing the initial
 *                   guess for the transformation from the bundle frame to the target frame.
 * @param opts A BundleOptions object specifying the optimization parameters and settings.
 *             Defaults to an empty configuration if not provided.
 *
 * @return A ScheimpflugBundleResult object containing the optimized camera parameters,
 *         poses, and other relevant results of the bundle adjustment.
 */
ScheimpflugBundleResult optimize_bundle_scheimpflug(
    const std::vector<BundleObservation>& observations,
    const std::vector<ScheimpflugCamera<BrownConradyd>>& initial_cameras,
    const std::vector<Eigen::Affine3d>& init_g_T_c,
    const Eigen::Affine3d& init_b_T_t,
    const BundleOptions& opts = {});

}  // namespace calib
