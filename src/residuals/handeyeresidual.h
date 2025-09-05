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
struct AxXbResidual final {
    Eigen::Matrix3d rot_a_, rot_b_;
    Eigen::Vector3d tra_a_, tra_b_;

    explicit AxXbResidual(const MotionPair& mp)
        : rot_a_(mp.rot_a), rot_b_(mp.rot_b), tra_a_(mp.tra_a), tra_b_(mp.tra_b) {}

    template <typename T>
    bool operator()(const T* const q, const T* const t, T* residuals) const {
        using Vec3 = Eigen::Matrix<T, 3, 1>;
        using Mat3 = Eigen::Matrix<T, 3, 3>;
        // q = [w, x, y, z], t = [tx, ty, tz]
        const Mat3 rot_x = quat_array_to_rotmat(q);
        const Mat3 rot_a = rot_a_.cast<T>();
        const Mat3 rot_b = rot_b_.cast<T>();
        const Mat3 rot_s = rot_a * rot_x * rot_b.transpose() * rot_x.transpose();
        Eigen::AngleAxis<T> axisangle(rot_s);

        // trans residual from (rot_a - I) tra_x = rot_x tra_b - tra_a
        const Vec3 tra_x(t[0], t[1], t[2]);
        const Vec3 tra_a = tra_a_.cast<T>();
        const Vec3 tra_b = tra_b_.cast<T>();
        const Vec3 tra_e = (rot_a - Mat3::Identity()) * tra_x - (rot_x * tra_b - tra_a);

        residuals[0] = axisangle.angle() * axisangle.axis()(0);
        residuals[1] = axisangle.angle() * axisangle.axis()(1);
        residuals[2] = axisangle.angle() * axisangle.axis()(2);
        residuals[3] = tra_e(0);
        residuals[4] = tra_e(1);
        residuals[5] = tra_e(2);
        return true;
    }

    static auto create(const MotionPair& mp) {
        return new ceres::AutoDiffCostFunction<AxXbResidual, 6, 4, 3>(new AxXbResidual(mp));
    }
};

}  // namespace calib
