/** @brief Utility functions for working with image observations */

#pragma once

// ceres
#include "ceres/rotation.h"

#include "calibration/planarpose.h"

namespace vitavision {

template<typename T>
Observation<T> to_observation(const PlanarObservation& obs, const T* pose6) {
    const T* aa = pose6;      // angle-axis
    const T* t  = pose6 + 3;  // translation

    Eigen::Matrix<T, 3, 1> P {T(obs.object_xy.x()), T(obs.object_xy.y()), T(0.0)};
    Eigen::Matrix<T, 3, 1> Pc;
    ceres::AngleAxisRotatePoint(aa, P.data(), Pc.data());
    Pc += Eigen::Matrix<T, 3, 1>(t[0], t[1], t[2]);
    T invZ = T(1.0) / Pc.z();
    Observation<T> ob;
    ob.x = Pc.x() * invZ;
    ob.y = Pc.y() * invZ;
    ob.u = T(obs.image_uv.x());
    ob.v = T(obs.image_uv.y());
    return ob;
}

// Utility: average a set of affine transforms (rotation via quaternion averaging)
inline Eigen::Affine3d average_affines(const std::vector<Eigen::Affine3d>& poses) {
    if (poses.empty()) return Eigen::Affine3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_sum(0,0,0,0);
    for (const auto& p : poses) {
        t += p.translation();
        Eigen::Quaterniond q(p.linear());
        if (q_sum.coeffs().dot(q.coeffs()) < 0.0) q.coeffs() *= -1.0;
        q_sum.coeffs() += q.coeffs();
    }
    t /= static_cast<double>(poses.size());
    q_sum.normalize();
    Eigen::Affine3d avg = Eigen::Affine3d::Identity();
    avg.linear() = q_sum.toRotationMatrix();
    avg.translation() = t;
    return avg;
}

// Utility: skew-symmetric matrix from vector
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<    0, -v.z(),  v.y(),
         v.z(),     0, -v.x(),
        -v.y(),  v.x(),    0;
    return m;
}

}  // namespace vitavision
