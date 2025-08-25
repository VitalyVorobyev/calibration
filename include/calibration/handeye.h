/** @brief Linear solvers for the hand-eye problem */

#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calibration/planarpose.h"

namespace vitavision {

Eigen::Affine3d estimate_hand_eye_tsai_lenz_allpairs_weighted(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg = 1.0);

struct RefinementOptions final {
    double huber_delta = 1.0;       // robust loss for residuals (radians & meters in same block)
    int max_iterations = 50;
    bool verbose = false;
};

Eigen::Affine3d refine_hand_eye(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    const Eigen::Affine3d& init_gripper_T_ref,
    const RefinementOptions& opts = {});

// ---------- convenience: full pipeline ----------
Eigen::Affine3d estimate_and_refine_hand_eye(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg = 1.0,
    const RefinementOptions& ro = {});

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

struct HandEyeReprojectionResult final {
    Eigen::Affine3d r_T_g;  // reference camera to gripper (hand-eye pose)
    Intrinsics intr;
    Eigen::Affine3d b_T_t;  // target to base
    double reprojection_error_pix = 0;
    Eigen::MatrixXd cov;
    std::string report;
};

/**
 * Refine hand-eye X (= ^gT_c), intrinsics, and ^bT_t by minimizing reprojection error.
 */
HandEyeReprojectionResult refine_hand_eye_reprojection(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<PlanarView> observables,
    const Intrinsics& init_intr,
    const Eigen::Affine3d& init_gripper_T_ref,
    const ReprojRefineOptions& options
);

}  // namespace vitavision
