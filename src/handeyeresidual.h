/** @brief Ceres residuals for hand-eye optimization
 * Three residuals are defined:
 * 1. AX = XB (rotation + translation)
 * 2. Reprojection (per view)
 */

#pragma once

// eigen
#include <Eigen/Geometry>

#include "calibration/planarpose.h"

#include "observationutils.h"

namespace vitavision {

struct MotionPair final {
    Eigen::Matrix3d RA, RB;
    Eigen::Vector3d tA, tB;
    double sqrt_weight; // sqrt of weight used to scale rows/residuals
};

// ---------- Ceres residual (AX = XB): rotation log + translation eq ----------
struct AX_XBResidual {
    Eigen::Matrix3d RA_, RB_;
    Eigen::Vector3d tA_, tB_;
    double sqrt_w_;

    AX_XBResidual(const MotionPair& mp) :
        RA_(mp.RA), RB_(mp.RB), tA_(mp.tA), tB_(mp.tB), sqrt_w_(mp.sqrt_weight) {}

    template <typename T>
    bool operator()(const T* const q, const T* const t, T* residuals) const {
        // q = [w, x, y, z], t = [tx, ty, tz]
        const Eigen::Matrix<T,3,3> RX = quat_array_to_rotmat(q);
        const Eigen::Matrix<T,3,3> RA = RA_.cast<T>();
        const Eigen::Matrix<T,3,3> RB = RB_.cast<T>();
        const Eigen::Matrix<T,3,3> RS = RA * RX * RB.transpose() * RX.transpose();
        Eigen::AngleAxis<T> aa(RS);

        // trans residual from (RA - I) tX = RX tB - tA
        const Eigen::Matrix<T,3,1> tX(t[0], t[1], t[2]);
        const Eigen::Matrix<T,3,1> tA = tA_.cast<T>();
        const Eigen::Matrix<T,3,1> tB = tB_.cast<T>();
        const Eigen::Matrix<T,3,1> et = (RA - Eigen::Matrix<T,3,3>::Identity()) * tX - (RX * tB - tA);

        const T s = T(sqrt_w_);
        residuals[0] = s * aa.angle() * aa.axis()(0);
        residuals[1] = s * aa.angle() * aa.axis()(1);
        residuals[2] = s * aa.angle() * aa.axis()(2);
        residuals[3] = s * et(0);
        residuals[4] = s * et(1);
        residuals[5] = s * et(2);
        return true;
    }

    static auto create(const MotionPair& mp) {
        return new ceres::AutoDiffCostFunction<AX_XBResidual, 6, 4, 3>(new AX_XBResidual(mp));
    }
};

// ---------- Reprojection residual (per view) ----------
struct HandEyeViewReprojResidual final {
    // Constants for this residual
    const Eigen::Matrix3d b_R_g_;   // ^bR_g at frame k
    const Eigen::Vector3d b_t_g_;   // ^bt_g at frame k
    const PlanarView view_;
    double sqrt_w_;
    const bool use_distortion_;

    HandEyeViewReprojResidual(const Eigen::Affine3d& base_T_gripper_k,
                   const PlanarView& view,
                   double weight = 1.0,
                   bool use_dist=false)
        : b_R_g_(base_T_gripper_k.linear()),
          b_t_g_(base_T_gripper_k.translation()),
          view_(view),
          sqrt_w_(std::sqrt(weight)),
          use_distortion_(use_dist) {}

    // Parameter blocks:
    // qX[4], tX[3] : hand-eye X = ^gT_c (camera in gripper)
    // qBt[4], tBt[3] : ^bT_t (target in base) -- assumed static across frames
    // intr[9] : [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    template <typename T>
    bool operator()(const T* const g_q_c, const T* const g_t_c,
                    const T* const b_q_t, const T* const b_t_t,
                    const T* const intr,
                    T* residuals) const
    {
        // Invert ^bT_g (constant per-residual)
        Eigen::Matrix<T,3,3> b_R_g = b_R_g_.cast<T>();
        Eigen::Matrix<T,3,1> b_t_g_vec = b_t_g_.cast<T>();
        auto [g_R_b, g_t_b_vec] = invert_transform(b_R_g, b_t_g_vec);

        // Invert X = ^gT_c
        Eigen::Matrix<T,3,3> g_R_c = quat_array_to_rotmat(g_q_c) ;  // ^gR_c
        Eigen::Matrix<T,3,1> g_t_c_vec = Eigen::Matrix<T,3,1>(g_t_c[0], g_t_c[1], g_t_c[2]);
        auto [c_R_g, c_t_g_vec] = invert_transform(g_R_c, g_t_c_vec);

        // Compose U = (g_T_b) * (b_T_t)
        Eigen::Matrix<T,3,3> b_R_t = quat_array_to_rotmat(b_q_t) ; // ^bR_t
        Eigen::Matrix<T,3,1> b_t_t_vec = Eigen::Matrix<T,3,1>(b_t_t[0], b_t_t[1], b_t_t[2]);
        auto [g_R_t, g_t_t_vec] = product(g_R_b, g_t_b_vec, b_R_t, b_t_t_vec);
        auto [c_R_t, c_t_t_vec] = product(c_R_g, c_t_g_vec, g_R_t, g_t_t_vec);  // c_T_t = c_T_g * g_T_t

        // Transform 3D point into camera frame and set residuals
        size_t idx = 0;
        T u_hat, v_hat;
        const T s = T(sqrt_w_);
        for (const auto& ob : view_) {
            Eigen::Matrix<T,3,1> P = Eigen::Matrix<T,3,1>(
                T(ob.object_xy.x()), T(ob.object_xy.y()), T(0));
            P = c_R_t * P + c_t_t_vec;
            project_with_intrinsics(P(0), P(1), P(2), intr, use_distortion_, u_hat, v_hat);
            residuals[idx++] = s * (u_hat - T(ob.image_uv.x()));
            residuals[idx++] = s * (v_hat - T(ob.image_uv.y()));
        }
        return true;
    }

    static auto create(const Eigen::Affine3d& base_T_gripper_k,
                       const PlanarView& view,
                       double weight = 1.0,
                       bool use_dist=false) {
        auto functor = new HandEyeViewReprojResidual(base_T_gripper_k, view, weight, use_dist);
        auto* cost = new ceres::AutoDiffCostFunction<HandEyeViewReprojResidual, ceres::DYNAMIC, 4, 3, 4, 3, 9>(
            functor, static_cast<int>(2 * view.size()));
        return cost;
    }
};

}  // namespace vitavision
