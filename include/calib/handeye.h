/** @brief Linear solvers for the hand-eye problem */

#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/planarpose.h"

namespace calib {

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
 * @param base_T_gripper A vector of transformations representing the motion
 *        of the gripper relative to the robot base. Each transformation is
 *        an Eigen::Affine3d object.
 * @param camera_T_target A vector of transformations representing the motion
 *        of the calibration target relative to the camera. Each transformation
 *        is an Eigen::Affine3d object.
 * @param min_angle_deg The minimum angular difference (in degrees) between
 *        two transformations to be considered for pairing. This parameter
 *        helps to filter out pairs with insufficient motion, which may lead
 *        to numerical instability. Default value is 1.0 degrees.
 * @return The estimated hand-eye transformation as an Eigen::Affine3d object.
 *
 * @note The input vectors `base_T_gripper` and `camera_T_target` must have
 *       the same size, and each pair of corresponding transformations must
 *       represent the same motion in the respective coordinate frames.
 * @throws std::invalid_argument If the input vectors have different sizes or
 *         if there are insufficient valid pairs for the estimation.
 */
Eigen::Affine3d estimate_handeye_dlt(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg = 1.0);

struct RefinementOptions final {
    double huber_delta = 1.0; // robust loss for residuals (radians & meters in same block)
    int max_iterations = 50;
    bool verbose = false;
};

/**
 * @brief Refines the hand-eye calibration transformation using an optimization process.
 *
 * This function takes a set of transformations representing the motion of the gripper
 * relative to the robot base and the motion of the calibration target relative to the camera.
 * It refines the initial estimate of the hand-eye transformation using the provided options.
 *
 * @param base_T_gripper A vector of transformations representing the motion of the gripper
 *                       relative to the robot base. Each transformation corresponds to a
 *                       specific pose of the gripper.
 * @param camera_T_target A vector of transformations representing the motion of the calibration
 *                        target relative to the camera. Each transformation corresponds to a
 *                        specific pose of the target.
 * @param init_gripper_T_ref The initial estimate of the transformation from the gripper to the
 *                           reference frame (hand-eye transformation).
 * @param opts Optional refinement options that control the optimization process. Defaults to
 *             an empty set of options.
 * @return The refined hand-eye transformation as an Eigen::Affine3d object.
 */
Eigen::Affine3d optimize_handeye(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    const Eigen::Affine3d& init_gripper_T_ref,
    const RefinementOptions& opts = {});


/**
 * @brief Estimates and refines the hand-eye transformation.
 *
 * This function computes the hand-eye transformation matrix that relates the
 * coordinate frame of a robotic gripper (end-effector) to the coordinate frame
 * of a camera mounted on the gripper. The estimation is performed using the
 * provided transformations, and an optional refinement step can be applied.
 *
 * @param base_T_gripper A vector of transformations representing the pose of
 *        the gripper relative to the robot base for multiple measurements.
 * @param camera_T_target A vector of transformations representing the pose of
 *        the calibration target relative to the camera for the same measurements.
 * @param min_angle_deg The minimum angular threshold (in degrees) to filter
 *        out small rotations during the refinement process. Default is 1.0 degrees.
 * @param ro Options for the refinement process, encapsulated in a
 *        RefinementOptions structure. Default is an empty RefinementOptions object.
 *
 * @return The estimated hand-eye transformation as an Eigen::Affine3d object.
 */
Eigen::Affine3d estimate_and_refine_hand_eye(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg = 1.0,
    const RefinementOptions& ro = {});

}  // namespace calib
