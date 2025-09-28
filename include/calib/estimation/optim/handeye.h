/**
 * @file handeye.h
 * @brief Hand-eye calibration algorithms and utilities
 * @ingroup hand_eye_calibration
 *
 * This file provides non-linear optimization interfaces for hand-eye calibration
 */

#pragma once

#include "calib/estimation/linear/handeye.h"
#include "calib/estimation/optim/optimize.h"

namespace calib {

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
