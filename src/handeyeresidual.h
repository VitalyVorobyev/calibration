/** @brief Ceres residuals for hand-eye optimization
 * Three residuals are defined:
 * 1. AX = XB (rotation + translation)
 * 2. Reprojection (per observation)
 * 3. (Optional) Additional residuals can be added as needed
 */

#pragma once

// eigen
#include <Eigen/Geometry>

#include "calibration/planarpose.h"

namespace vitavision {

struct MotionPair final {
    Eigen::Matrix3d RA, RB;
    Eigen::Vector3d tA, tB;
    double sqrt_weight; // sqrt of weight used to scale rows/residuals
};

template<typename T>
static Eigen::Matrix<T,3,3> quat_array_to_rotmat(const T* const arr) {
    Eigen::Quaternion<T> q(arr[0], arr[1], arr[2], arr[3]);
    return q.toRotationMatrix();
}

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
        const Eigen::Matrix<T,3,3> RS = RA * RX * RB.transpose() * RXT.transpose();
        Eigen::AngleAxis<T> aa(RS);

        #if 0
        // rot residual: angle-axis of RS
        T RS_cols[9] = {
            RS(0,0), RS(1,0), RS(2,0),
            RS(0,1), RS(1,1), RS(2,1),
            RS(0,2), RS(1,2), RS(2,2)
        };
        T aa[3];
        ceres::RotationMatrixToAngleAxis(RS_cols, aa);
        #endif

        // trans residual from (RA - I) tX = RX tB - tA
        const Eigen::Matrix<T,3,1> tX(t[0], t[1], t[2]);
        const Eigen::Matrix<T,3,1> tA = tA_.cast<T>();
        const Eigen::Matrix<T,3,1> tB = tB_.cast<T>();
        const Eigen::Matrix<T,3,1> et = (RA - Eigen::Matrix<T,3,3>::Identity()) * tX - (RX * tB - tA);

        const T s = T(sqrt_w_);
        residuals[0] = s * aa(0);
        residuals[1] = s * aa(1);
        residuals[2] = s * aa(2);
        residuals[3] = s * et(0);
        residuals[4] = s * et(1);
        residuals[5] = s * et(2);
        return true;
    }

    static auto create(const MotionPair& mp) {
        return new ceres::AutoDiffCostFunction<AX_XBResidual, 6, 4, 3>(new AX_XBResidual(mp));
    }
};

#if 0
// ---------- Reprojection residual (per observation) ----------
struct HandEyeReprojResidual final {
    // Constants for this residual
    const Eigen::Matrix3d b_R_g;   // ^bR_g at frame k
    const Eigen::Vector3d b_t_g;   // ^bt_g at frame k
    const Eigen::Vector3d P_t;    // target 3D point in target frame
    const double u_obs, v_obs;    // observed pixel
    const double sqrt_w;          // optional weight (e.g., 1.0)
    const bool use_distortion;

    HandEyeReprojResidual(const Eigen::Affine3d& base_T_gripper_k,
                   const Eigen::Vector3d& Pt,
                   const Eigen::Vector2d& uv,
                   double weight = 1.0,
                   bool use_dist=false)
        : b_R_g(base_T_gripper_k.linear()),
          b_t_g(base_T_gripper_k.translation()),
          P_t(Pt),
          u_obs(uv.x()),
          v_obs(uv.y()),
          sqrt_w(std::sqrt(weight)),
          use_distortion(use_dist) {}

    // Parameter blocks:
    // qX[4], tX[3] : hand-eye X = ^gT_c (camera in gripper)
    // qBt[4], tBt[3] : ^bT_t (target in base) -- assumed static across frames
    // intr[9] : [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    template <typename T>
    bool operator()(const T* const qX, const T* const tX,
                    const T* const qBt, const T* const tBt,
                    const T* const intr,
                    T* residuals) const
    {
        // Convert quaternions to rotation matrices
        T RX_raw[9], RBt_raw[9];
        ceres::QuaternionToRotation(qX, RX_raw);
        ceres::QuaternionToRotation(qBt, RBt_raw);

        Eigen::Matrix<T,3,3> RX;  // ^gR_c
        RX << RX_raw[0], RX_raw[3], RX_raw[6],
              RX_raw[1], RX_raw[4], RX_raw[7],
              RX_raw[2], RX_raw[5], RX_raw[8];

        Eigen::Matrix<T,3,3> RBt; // ^bR_t
        RBt << RBt_raw[0], RBt_raw[3], RBt_raw[6],
               RBt_raw[1], RBt_raw[4], RBt_raw[7],
               RBt_raw[2], RBt_raw[5], RBt_raw[8];

        // Invert ^bT_g (constant per-residual)
        Eigen::Matrix<T,3,3> Rbg = b_R_g.cast<T>();
        Eigen::Matrix<T,3,3> Rgb = Rbg.transpose();
        Eigen::Matrix<T,3,1> tbg = b_t_g.cast<T>();
        Eigen::Matrix<T,3,1> tgb = -Rgb * tbg;

        // Invert X = ^gT_c
        Eigen::Matrix<T,3,3> Rcx = RX.transpose();         // ^cR_g
        Eigen::Matrix<T,3,1> tx  = Eigen::Matrix<T,3,1>(tX[0], tX[1], tX[2]);
        Eigen::Matrix<T,3,1> tcg = -Rcx * tx;              // ^ct_g

        // Compose U = (^bT_g)^{-1} * (^bT_t)
        Eigen::Matrix<T,3,1> tbt = Eigen::Matrix<T,3,1>(tBt[0], tBt[1], tBt[2]);
        Eigen::Matrix<T,3,3> Ru  = Rgb * RBt;
        Eigen::Matrix<T,3,1> tu  = Rgb * (tbt - tbg);

        // ^cT_t = X^{-1} * U
        Eigen::Matrix<T,3,3> Rct = Rcx * Ru;
        Eigen::Matrix<T,3,1> tct = Rcx * tu + tcg;

        // Transform 3D point into camera frame
        Eigen::Matrix<T,3,1> Pt = P_t.cast<T>();
        Eigen::Matrix<T,3,1> Pc = Rct * Pt + tct;

        // Project
        T u_hat, v_hat;
        project_with_intrinsics(Pc(0), Pc(1), Pc(2), intr, use_distortion, u_hat, v_hat);

        // Residuals
        const T s = T(sqrt_w);
        residuals[0] = s * (u_hat - T(u_obs));
        residuals[1] = s * (v_hat - T(v_obs));
        return true;
    }

    static auto create(const Eigen::Affine3d& base_T_gripper_k,
                   const Eigen::Vector3d& Pt,  // target point
                   const Eigen::Vector2d& uv,  // image point
                   double weight = 1.0,
                   bool use_dist=false) {
        auto* cost = new ceres::AutoDiffCostFunction<HandEyeReprojResidual, 2, 4, 3, 4, 3, 9>(
            new HandEyeReprojResidual(base_T_gripper_k, Pt, uv, weight, use_dist)
        );
        return cost;
    }
};
#endif

template<typename T>
std::pair<Eigen::Matrix<T,3,3>, Eigen::Matrix<T,3,1>> invert_transform(
    const Eigen::Matrix<T,3,3>& R,
    const Eigen::Matrix<T,3,1>& t
) {
    Eigen::Matrix<T,3,3> Rt = R.transpose();
    Eigen::Matrix<T,3,1> ti = -Rt * t;
    return {Rt, ti};
}

template<typename T>
std::pair<Eigen::Matrix<T,3,3>, Eigen::Matrix<T,3,1>> product(
    const Eigen::Matrix<T,3,3>& R1, const Eigen::Matrix<T,3,1>& t1,
    const Eigen::Matrix<T,3,3>& R2, const Eigen::Matrix<T,3,1>& t2
) {
    Eigen::Matrix<T,3,3> R = R1 * R2;
    Eigen::Matrix<T,3,1> t = R1 * t2 + t1;
    return {R, t};
}

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
            Eigen::Matrix<T,3,1> P = Eigen::Matrix<T,3,1>(T(ob.object_xy.x()), T(ob.object_xy.y()), T(0));
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

// Optional: soft AX=XB prior (rotation log + translation equation), weight lambda
struct AXXBResidualSoft final {
    const Eigen::Matrix3d RA_, RB_;
    const Eigen::Vector3d tA_, tB_;
    const double sqrt_w_;

    AXXBResidualSoft(const Eigen::Matrix3d& RA, const Eigen::Matrix3d& RB,
                     const Eigen::Vector3d& tA, const Eigen::Vector3d& tB,
                     double lambda)
        : RA_(RA), RB_(RB), tA_(tA), tB_(tB), sqrt_w_(std::sqrt(lambda)) {}

    template <typename T>
    bool operator()(const T* const qX, const T* const tX, T* residuals) const {
        T RX_raw[9];
        ceres::QuaternionToRotation(qX, RX_raw);
        Eigen::Matrix<T,3,3> RX;
        RX << RX_raw[0], RX_raw[3], RX_raw[6],
              RX_raw[1], RX_raw[4], RX_raw[7],
              RX_raw[2], RX_raw[5], RX_raw[8];

        Eigen::Matrix<T,3,3> RA = RA_.cast<T>();
        Eigen::Matrix<T,3,3> RB = RB_.cast<T>();
        Eigen::Matrix<T,3,3> RS = RA * RX * RB * RX.transpose();

        // rotation part -> angle-axis
        T RS_cols[9] = {
            RS(0,0), RS(1,0), RS(2,0),
            RS(0,1), RS(1,1), RS(2,1),
            RS(0,2), RS(1,2), RS(2,2)
        };
        T aa[3];
        ceres::RotationMatrixToAngleAxis(RS_cols, aa);

        // translation part
        const Eigen::Matrix<T,3,1> tXv(tX[0], tX[1], tX[2]);
        const Eigen::Matrix<T,3,1> tA = tA_.cast<T>(), tB = tB_.cast<T>();
        Eigen::Matrix<T,3,1> et = (RA - Eigen::Matrix<T,3,3>::Identity())*tXv - (RX * tB - tA);

        const T s = T(sqrt_w_);
        residuals[0] = s * aa[0];
        residuals[1] = s * aa[1];
        residuals[2] = s * aa[2];
        residuals[3] = s * et(0);
        residuals[4] = s * et(1);
        residuals[5] = s * et(2);
        return true;
    }

    static auto create(const Eigen::Matrix3d& RA, const Eigen::Matrix3d& RB,
                     const Eigen::Vector3d& tA, const Eigen::Vector3d& tB,
                     double lambda) {
        auto* cost = new ceres::AutoDiffCostFunction<AXXBResidualSoft, 6, 4, 3>(
            new AXXBResidualSoft(RA, RB, tA, tB, lambda)
        );
        return cost;
    }
};

}  // namespace vitavision
