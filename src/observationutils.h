/** @brief Utility functions for working with image observations */

#pragma once

// ceres
#include "calib/distortion.h"
#include "calib/planarpose.h"
#include "ceres/rotation.h"

namespace calib {

// templated for autodiff
template <typename T>
static Eigen::Matrix<T, 3, 1> array_to_translation(const T* const arr) {
    return Eigen::Matrix<T, 3, 1>(arr[0], arr[1], arr[2]);
}

template <typename T>
static Eigen::Matrix<T, 3, 3> quat_array_to_rotmat(const T* const arr) {
    Eigen::Quaternion<T> q(arr[0], arr[1], arr[2], arr[3]);
    return q.toRotationMatrix();
}

template <typename T>
std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> invert_transform(
    const Eigen::Matrix<T, 3, 3>& R, const Eigen::Matrix<T, 3, 1>& t) {
    Eigen::Matrix<T, 3, 3> Rt = R.transpose();
    Eigen::Matrix<T, 3, 1> ti = -Rt * t;
    return {Rt, ti};
}

template <typename T>
std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> product(
    const Eigen::Matrix<T, 3, 3>& R1, const Eigen::Matrix<T, 3, 1>& t1,
    const Eigen::Matrix<T, 3, 3>& R2, const Eigen::Matrix<T, 3, 1>& t2) {
    Eigen::Matrix<T, 3, 3> R = R1 * R2;
    Eigen::Matrix<T, 3, 1> t = R1 * t2 + t1;
    return {R, t};
}

inline void populate_quat_tran(const Eigen::Isometry3d& pose, std::array<double, 4>& q,
                               std::array<double, 3>& t) {
    Eigen::Quaterniond q0(pose.linear());
    q = {q0.w(), q0.x(), q0.y(), q0.z()};
    t = {pose.translation().x(), pose.translation().y(), pose.translation().z()};
}

inline Eigen::Quaterniond array_to_norm_quat(const std::array<double, 4>& arr) {
    Eigen::Quaterniond quat(arr[0], arr[1], arr[2], arr[3]);
    quat.normalize();
    return quat;
}

inline Eigen::Isometry3d restore_pose(const std::array<double, 4>& q,
                                    const std::array<double, 3>& t) {
    auto pose = Eigen::Isometry3d::Identity();
    pose.linear() = array_to_norm_quat(q).toRotationMatrix();
    pose.translation() << t[0], t[1], t[2];
    return pose;
}

// Utility: convert rotation+translation to inverse transform quickly
inline void invertRT(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, Eigen::Matrix3d& Rinv,
                     Eigen::Vector3d& tinv) {
    Rinv = R.transpose();
    tinv = -Rinv * t;
}

// ---------- small SO(3) helpers (double) ----------
inline Eigen::Matrix3d projectToSO3(const Eigen::Matrix3d& R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    if ((U * V.transpose()).determinant() < 0.0) S(2, 2) = -1.0;
    return U * S * V.transpose();
}

// Utility: skew-symmetric matrix from vector
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return S;
}

// log(R) as a 3-vector (axis*angle)
inline Eigen::Vector3d logSO3(const Eigen::Matrix3d& R_in) {
    const Eigen::Matrix3d R = projectToSO3(R_in);
    double cos_theta = (R.trace() - 1.0) * 0.5;
    cos_theta = std::min(1.0, std::max(-1.0, cos_theta));
    double theta = std::acos(cos_theta);
    if (theta < 1e-12) return Eigen::Vector3d::Zero();
    Eigen::Vector3d w;
    w << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
    w *= 0.5 / std::sin(theta);
    return w * theta;
}

inline Eigen::Matrix3d expSO3(const Eigen::Vector3d& w) {
    double theta = w.norm();
    if (theta < 1e-12) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d a = w / theta;
    Eigen::Matrix3d A = skew(a);
    return Eigen::Matrix3d::Identity() + std::sin(theta) * A + (1.0 - std::cos(theta)) * (A * A);
}

inline Eigen::VectorXd solve_llsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    return A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}

// ---------- stable ridge LS solve ----------
template <class Mat, class Vec>
Eigen::VectorXd ridge_llsq(const Mat& A, const Vec& b, double lambda = 1e-10) {
    const int p = static_cast<int>(A.cols());
    return (A.transpose() * A + lambda * Eigen::MatrixXd::Identity(p, p))
        .ldlt()
        .solve(A.transpose() * b);
}

inline Eigen::Vector3d log_rot(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd aa(R);
    return aa.axis() * aa.angle();
}

inline std::array<double, 6> pose_to_array(const Eigen::Isometry3d& pose) {
    Eigen::AngleAxisd aa(pose.linear());
    return {aa.axis().x() * aa.angle(), aa.axis().y() * aa.angle(), aa.axis().z() * aa.angle(),
            pose.translation().x(),     pose.translation().y(),     pose.translation().z()};
}

inline Eigen::Isometry3d array_to_pose(const double* p) {
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(p, R.data());
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = R;
    T.translation() = Eigen::Vector3d(p[3], p[4], p[5]);
    return T;
}

template <typename T>
Eigen::Transform<T, 3, Eigen::Isometry> pose2affine(const T* pose) {
    Eigen::Matrix<T, 3, 3> R;
    ceres::AngleAxisToRotationMatrix(pose, R.data());
    Eigen::Matrix<T, 3, 1> t{pose[3], pose[4], pose[5]};
    return Eigen::Translation<T, 3>(t) * R;
}

inline Eigen::Isometry3d pose6_to_affine(const Eigen::VectorXd& p) {
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(p.head<3>().data(), R.data());
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = R;
    T.translation() = p.tail<3>();
    return T;
}

// Utility: average a set of affine transforms (rotation via quaternion averaging)
inline Eigen::Isometry3d average_affines(const std::vector<Eigen::Isometry3d>& poses) {
    if (poses.empty()) return Eigen::Isometry3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_sum(0, 0, 0, 0);
    for (const auto& p : poses) {
        t += p.translation();
        Eigen::Quaterniond q(p.linear());
        if (q_sum.coeffs().dot(q.coeffs()) < 0.0) q.coeffs() *= -1.0;
        q_sum.coeffs() += q.coeffs();
    }
    t /= static_cast<double>(poses.size());
    q_sum.normalize();
    Eigen::Isometry3d avg = Eigen::Isometry3d::Identity();
    avg.linear() = q_sum.toRotationMatrix();
    avg.translation() = t;
    return avg;
}

template <typename T>
static void planar_observables_to_observables(
    const PlanarView& po, std::vector<Observation<T>>& o,
    const Eigen::Transform<T, 3, Eigen::Isometry>& camera_se3_target) {
    if (o.size() != po.size()) o.resize(po.size());
    for (size_t i = 0; i < po.size(); ++i) {
        const auto& p = po[i];
        // Convert pixel coordinates to normalized image coordinates
        Eigen::Matrix<T, 3, 1> P{T(p.object_xy.x()), T(p.object_xy.y()), T(0)};
        P = camera_se3_target * P;
        const T xn = P.x() / P.z();
        const T yn = P.y() / P.z();
        o[i] = Observation<T>{xn, yn, T(p.image_uv.x()), T(p.image_uv.y())};
    }
}

template <typename T>
Observation<T> to_observation(const PlanarObservation& obs, const T* pose6) {
    const T* aa = pose6;     // angle-axis
    const T* t = pose6 + 3;  // translation

    Eigen::Matrix<T, 3, 1> P{T(obs.object_xy.x()), T(obs.object_xy.y()), T(0.0)};
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

}  // namespace calib
