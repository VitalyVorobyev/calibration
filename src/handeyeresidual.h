/** @brief Ceres residuals for hand-eye optimization */

#pragma once

// eigen
#include <Eigen/Geometry>

namespace vitavision {

// ---------- Reprojection residual (per observation) ----------
struct ReprojResidual final {
    // Constants for this residual
    Eigen::Matrix3d R_bg;   // ^bR_g at frame k
    Eigen::Vector3d t_bg;   // ^bt_g at frame k
    Eigen::Vector3d P_t;    // target 3D point in target frame
    double u_obs, v_obs;    // observed pixel
    double sqrt_w;          // optional weight (e.g., 1.0)
    bool use_distortion;

    ReprojResidual(const Eigen::Affine3d& base_T_gripper_k,
                   const Eigen::Vector3d& Pt,
                   const Eigen::Vector2d& uv,
                   double weight = 1.0,
                   bool use_dist=false)
        : R_bg(base_T_gripper_k.linear()),
          t_bg(base_T_gripper_k.translation()),
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
        Eigen::Matrix<T,3,3> Rbg = R_bg.cast<T>();
        Eigen::Matrix<T,3,3> Rgb = Rbg.transpose();
        Eigen::Matrix<T,3,1> tbg = t_bg.cast<T>();
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
};

// Optional: soft AX=XB prior (rotation log + translation equation), weight lambda
struct AXXBResidualSoft {
    Eigen::Matrix3d RA_, RB_;
    Eigen::Vector3d tA_, tB_;
    double sqrt_w_;

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
};

}  // namespace vitavision
