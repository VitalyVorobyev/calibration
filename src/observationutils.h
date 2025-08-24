/** @brief Utility functions for working with image observations */

#pragma once

// ceres
#include "ceres/rotation.h"

#include "calibration/planarpose.h"

namespace vitavision {

using Pose6 = Eigen::Matrix<double, 6, 1>;

inline Eigen::VectorXd solve_llsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    return A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}

inline Eigen::Vector3d log_rot(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd aa(R);
    return aa.axis() * aa.angle();
}

inline std::array<double, 6> pose_to_array(const Eigen::Affine3d& pose) {
    Eigen::AngleAxisd aa(pose.linear());
    return {
        aa.axis().x() * aa.angle(),
        aa.axis().y() * aa.angle(),
        aa.axis().z() * aa.angle(),
        pose.translation().x(),
        pose.translation().y(),
        pose.translation().z()
    };
}

inline Eigen::Affine3d array_to_pose(const double* p) {
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(p, R.data());
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear() = R;
    T.translation() = Eigen::Vector3d(p[3], p[4], p[5]);
    return T;
}

template<typename T>
Eigen::Transform<T, 3, Eigen::Affine> pose2affine(const T* pose) {
    Eigen::Matrix<T, 3, 3> R;
    ceres::AngleAxisToRotationMatrix(pose, R.data());
    Eigen::Matrix<T, 3, 1> t{pose[3], pose[4], pose[5]};
    return Eigen::Translation<T, 3>(t) * R;
}

// Utility: skew-symmetric matrix from vector
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<    0, -v.z(),  v.y(),
         v.z(),     0, -v.x(),
        -v.y(),  v.x(),    0;
    return m;
}

inline Eigen::Affine3d pose6_to_affine(const Pose6& p) {
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(p.head<3>().data(), R.data());
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear() = R;
    T.translation() = p.tail<3>();
    return T;
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

template<typename T>
static void planar_observables_to_observables(
    const PlanarView& po,
    std::vector<Observation<T>>& o,
    const Eigen::Transform<T, 3, Eigen::Affine>& camera_T_target
) {
    if (o.size() != po.size()) o.resize(po.size());
    for (size_t i = 0; i < po.size(); ++i) {
        const auto& p = po[i];
        // Convert pixel coordinates to normalized image coordinates
        Eigen::Matrix<T,3,1> P{T(p.object_xy.x()), T(p.object_xy.y()), T(0)};
        P = camera_T_target * P;
        const T xn = P.x() / P.z();
        const T yn = P.y() / P.z();
        o[i] = Observation<T>{xn, yn, T(p.image_uv.x()), T(p.image_uv.y())};
    }
}

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

}  // namespace vitavision
