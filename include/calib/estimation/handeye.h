/**
 * @file handeye.h
 * @brief Hand-eye calibration algorithms and utilities
 * @ingroup hand_eye_calibration
 *
 * This file provides comprehensive hand-eye calibration functionality including:
 * - Tsai-Lenz algorithm for AX=XB problem solving
 * - Motion pair generation and filtering
 * - Bundle adjustment for hand-eye calibration
 * - Weighted averaging for robust estimation
 */

#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/estimation/planarpose.h"
#include "calib/estimation/optimize.h"

namespace calib {

/**
 * @brief Motion pair structure for hand-eye calibration
 * @ingroup hand_eye_calibration
 *
 * Represents a pair of corresponding motions between two coordinate frames:
 * - Motion A: typically robot base to gripper transformation
 * - Motion B: typically camera to target transformation
 *
 * Used in solving the AX=XB hand-eye calibration problem.
 */
struct MotionPair final {
    Eigen::Matrix3d rot_a, rot_b;  ///< Rotation matrices for motions A and B
    Eigen::Vector3d tra_a, tra_b;  ///< Translation vectors for motions A and B
};

/**
 * @brief Generate all valid motion pairs from pose sequences
 * @ingroup hand_eye_calibration
 *
 * Creates motion pairs from sequences of robot and camera poses,
 * filtering out motions that are too small or have parallel axes
 * to ensure numerical stability.
 *
 * @param base_se3_gripper Robot base to gripper transformations
 * @param cam_se3_target Camera to target transformations
 * @param min_angle_deg Minimum rotation angle to accept (degrees)
 * @param reject_axis_parallel Whether to reject parallel rotation axes
 * @param axis_parallel_eps Threshold for parallel axis detection
 * @return Vector of valid motion pairs
 */
auto build_all_pairs(const std::vector<Eigen::Isometry3d>& base_se3_gripper,
                     const std::vector<Eigen::Isometry3d>& cam_se3_target,
                     double min_angle_deg = 1.0,        // discard too-small motions
                     bool reject_axis_parallel = true,  // guard against ill-conditioning
                     double axis_parallel_eps = 1e-3) -> std::vector<MotionPair>;

/**
 * @brief Estimates the hand-eye transformation using the Tsai-Lenz algorithm
 *        with all pairs of input transformations and weighted averaging.
 *
 * This function computes the hand-eye transformation that relates the motion
 * of a robotic gripper (end-effector) to the motion of a camera mounted on the
 * gripper. It uses the Tsai-Lenz algorithm and considers all pairs of input
 * transformations, applying a weighting scheme to improve the robustness of
 * the estimation.
 *
 * @param base_se3_gripper A vector of transformations representing the motion
 *        of the gripper relative to the robot base. Each transformation is
 *        an Eigen::Isometry3d object.
 * @param camera_se3_target A vector of transformations representing the motion
 *        of the calibration target relative to the camera. Each transformation
 *        is an Eigen::Isometry3d object.
 * @param min_angle_deg The minimum angular difference (in degrees) between
 *        two transformations to be considered for pairing. This parameter
 *        helps to filter out pairs with insufficient motion, which may lead
 *        to numerical instability. Default value is 1.0 degrees.
 * @return The estimated hand-eye transformation as an Eigen::Isometry3d object.
 *
 * @note The input vectors `base_se3_gripper` and `camera_se3_target` must have
 *       the same size, and each pair of corresponding transformations must
 *       represent the same motion in the respective coordinate frames.
 * @throws std::invalid_argument If the input vectors have different sizes or
 *         if there are insufficient valid pairs for the estimation.
 */
auto estimate_handeye_dlt(const std::vector<Eigen::Isometry3d>& base_se3_gripper,
                          const std::vector<Eigen::Isometry3d>& camera_se3_target,
                          double min_angle_deg = 1.0) -> Eigen::Isometry3d;

struct HandeyeOptions final : public OptimOptions {};

struct HandeyeResult final : public OptimResult {
    Eigen::Isometry3d g_se3_c;  ///< Estimated hand-eye transform (gripper -> camera)
};

/**
 * @brief Refines the hand-eye calibration transformation using an optimization process.
 *
 * This function takes a set of transformations representing the motion of the gripper
 * relative to the robot base and the motion of the calibration target relative to the camera.
 * It refines the initial estimate of the hand-eye transformation using the provided options.
 *
 * @param base_se3_gripper A vector of transformations representing the motion of the gripper
 *                       relative to the robot base. Each transformation corresponds to a
 *                       specific pose of the gripper.
 * @param camera_se3_target A vector of transformations representing the motion of the calibration
 *                        target relative to the camera. Each transformation corresponds to a
 *                        specific pose of the target.
 * @param init_gripper_se3_ref The initial estimate of the transformation from the gripper to the
 *                           reference frame (hand-eye transformation).
 * @param options Optional refinement options that control the optimization process. Defaults to
 *                an empty set of options.
 * @return The refined hand-eye transformation as an Eigen::Isometry3d object.
 */
auto optimize_handeye(const std::vector<Eigen::Isometry3d>& base_se3_gripper,
                      const std::vector<Eigen::Isometry3d>& camera_se3_target,
                      const Eigen::Isometry3d& init_gripper_se3_ref,
                      const HandeyeOptions& options = {}) -> HandeyeResult;

/**
 * @brief Estimates and refines the hand-eye transformation.
 *
 * This function computes the hand-eye transformation matrix that relates the
 * coordinate frame of a robotic gripper (end-effector) to the coordinate frame
 * of a camera mounted on the gripper. The estimation is performed using the
 * provided transformations, and an optional refinement step can be applied.
 *
 * @param base_se3_gripper A vector of transformations representing the pose of
 *        the gripper relative to the robot base for multiple measurements.
 * @param camera_se3_target A vector of transformations representing the pose of
 *        the calibration target relative to the camera for the same measurements.
 * @param min_angle_deg The minimum angular threshold (in degrees) to filter
 *        out small rotations during the refinement process. Default is 1.0 degrees.
 * @param ro Options for the refinement process, encapsulated in a
 *        RefinementOptions structure. Default is an empty RefinementOptions object.
 *
 * @return The estimated hand-eye transformation as an Eigen::Isometry3d object.
 */
auto estimate_and_optimize_handeye(const std::vector<Eigen::Isometry3d>& base_se3_gripper,
                                   const std::vector<Eigen::Isometry3d>& camera_se3_target,
                                   double min_angle_deg = 1.0, const HandeyeOptions& options = {})
    -> HandeyeResult;

}  // namespace calib
