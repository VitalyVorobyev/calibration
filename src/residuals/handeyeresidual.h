/** @brief Ceres residuals for hand-eye optimization
 * Three residuals are defined:
 * 1. AX = XB (rotation + translation)
 * 2. Reprojection (per view)
 */

#pragma once

// eigen
#include <Eigen/Geometry>

#include "../observationutils.h"
#include "calib/handeye.h"  // MotionPair

namespace calib {

// ---------- Ceres residual (AX = XB): rotation log + translation eq ----------
struct AX_XBResidual final {
    Eigen::Matrix3d RA_, RB_;
    Eigen::Vector3d tA_, tB_;

    explicit AX_XBResidual(const MotionPair& mp) : RA_(mp.RA), RB_(mp.RB), tA_(mp.tA), tB_(mp.tB) {}

    template <typename T>
    bool operator()(const T* const q, const T* const t, T* residuals) const {
        using Vec3 = Eigen::Matrix<T, 3, 1>;
        using Mat3 = Eigen::Matrix<T, 3, 3>;
        // q = [w, x, y, z], t = [tx, ty, tz]
        const Mat3 RX = quat_array_to_rotmat(q);
        const Mat3 RA = RA_.cast<T>();
        const Mat3 RB = RB_.cast<T>();
        const Mat3 RS = RA * RX * RB.transpose() * RX.transpose();
        Eigen::AngleAxis<T> aa(RS);

        // trans residual from (RA - I) tX = RX tB - tA
        const Vec3 tX(t[0], t[1], t[2]);
        const Vec3 tA = tA_.cast<T>();
        const Vec3 tB = tB_.cast<T>();
        const Vec3 et = (RA - Mat3::Identity()) * tX - (RX * tB - tA);

        residuals[0] = aa.angle() * aa.axis()(0);
        residuals[1] = aa.angle() * aa.axis()(1);
        residuals[2] = aa.angle() * aa.axis()(2);
        residuals[3] = et(0);
        residuals[4] = et(1);
        residuals[5] = et(2);
        return true;
    }

    static auto create(const MotionPair& mp) {
        return new ceres::AutoDiffCostFunction<AX_XBResidual, 6, 4, 3>(new AX_XBResidual(mp));
    }
};

}  // namespace calib
