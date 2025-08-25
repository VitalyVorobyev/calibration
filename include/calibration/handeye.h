/** @brief Linear solvers for the hand-eye problem */

#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

namespace vitavision {

/**
 * Compute an initial estimate of the hand-eye transform (camera -> gripper)
 * using the Tsai-Lenz linear method.  The input vectors must contain
 * corresponding poses of the robot end-effector in the base frame and poses of
 * the planar target in the camera frame for the same time instants.
 * @param base_T_gripper A vector of affine transformations representing the poses
 *        of the robotic gripper in the base frame.
 * @param camera_T_target A vector of affine transformations representing the poses
 *        of the camera relative to the observed target.
 * @return The estimated hand-eye transform (camera -> gripper).
 */
Eigen::Affine3d estimate_hand_eye_tsai_lenz(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target);

Eigen::Affine3d estimate_hand_eye_tsai_lenz_allpairs_weighted(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg = 1.0);

struct RefinementOptions final {
    double huber_delta = 1.0;       // robust loss for residuals (radians & meters in same block)
    int max_iterations = 50;
    bool verbose = false;
};

Eigen::Affine3d refine_hand_eye_ceres(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    const Eigen::Affine3d& X0,
    const RefinementOptions& opts = {});

struct Intrinsics final {
    double fx, fy, cx, cy;                 // pinhole
    double k1=0, k2=0, p1=0, p2=0, k3=0;   // Brown 5 distortion
    bool use_distortion = false;
};

struct ReprojRefineOptions {
    bool refine_intrinsics = false;     // optimize fx,fy,cx,cy
    bool refine_distortion = false;     // optimize k1..k3, p1,p2 (requires refine_intrinsics = true)
    bool use_distortion = false;        // use distortion in projection
    double huber_delta_px = 1.0;        // robust loss on pixel residuals
    int max_iterations = 80;
    bool verbose = false;

    // Optional soft AX=XB prior (lambda=0 -> disabled)
    double lambda_axxb = 0.0;           // weight for AX=XB residuals
};

/**
 * Refine hand-eye X (= ^gT_c), intrinsics, and ^bT_t by minimizing reprojection error.
 *
 * @param base_T_gripper   per-frame ^bT_g
 * @param object_points    3D points in target frame (shared across frames)
 * @param image_points     per-frame 2D observations; image_points[k].size() must equal object_points.size()
 * @param intr             initial intrinsics
 * @param X0               initial hand-eye (e.g., Tsai–Lenz)
 * @param options          refinement options
 * @param out_intr         (output) refined intrinsics
 * @param out_b_T_t        (output) estimated target pose in base frame
 * @return refined X (= ^gT_c)
 */
Eigen::Affine3d refine_hand_eye_reprojection(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<std::vector<Eigen::Vector2d>>& image_points,
    const Intrinsics& intr,
    const Eigen::Affine3d& X0,
    const ReprojRefineOptions& options,
    Intrinsics& out_intr,
    Eigen::Affine3d& out_b_T_t
);

// ---------- convenience: full pipeline ----------
Eigen::Affine3d estimate_and_refine_hand_eye(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg = 1.0,
    const RefinementOptions& ro = {});

}  // namespace vitavision
