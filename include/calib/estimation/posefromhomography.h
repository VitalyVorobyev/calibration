/** @brief Planar target pose from homography */

#pragma once

// std
#include <string>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calib/core/cameramatrix.h"

namespace calib {

/**
 * @brief Recover target->camera pose (R,t) from camera matrix K and planar homography H.
 *
 * Assumptions:
 *  - H maps target planar homogeneous coords (X,Y,1) to pixel coords (u,v,1).
 *  - Distortion is negligible (or already compensated).
 *  - K is well-conditioned (fx, fy > 0).
 *
 * Steps:
 *  1) Hn = K^{-1} H
 *  2) scale s = 1 / mean(||Hn.col(0)||, ||Hn.col(1)||)
 *  3) r1 = s * Hn.col(0),  r2 = s * Hn.col(1),  t = s * Hn.col(2)
 *  4) r3 = r1 x r2;  R_pre = [r1 r2 r3]
 *  5) Orthonormalize R via SVD (polar decomposition): R = U V^T (det>0)
 *  6) Enforce positive forward direction (t_z > 0); if not, flip R,t
 */
struct PoseFromHResult final {
    bool success{false};
    Eigen::Isometry3d c_se3_t = Eigen::Isometry3d::Identity();  // camera_T_target
    double scale{1.0};                                          // internal homography scale s
    double cond_check{1.0};  // ratio of ||h1|| and ||h2|| (close to 1 is good)
    std::string message;
};

auto pose_from_homography(const CameraMatrix& kmtx, const Eigen::Matrix3d& hmtx) -> PoseFromHResult;

/**
 * @brief A quick consistency check. Returns ||K [R(:,1) R(:,2) t] - H||_F / ||H||_F.
 *        Values ~ 1e-3..1e-2 are typical when H was estimated from the same correspondences used.
 */
auto homography_consistency_fro(const CameraMatrix& kmtx, const Eigen::Isometry3d& c_se3_t,
                                const Eigen::Matrix3d& hmtx) -> double;

}  // namespace calib
