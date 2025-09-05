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
    Eigen::Quaternion<T> quat(arr[0], arr[1], arr[2], arr[3]);
    return quat.toRotationMatrix();
}

template <typename T>
std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> invert_transform(
    const Eigen::Matrix<T, 3, 3>& rotation, const Eigen::Matrix<T, 3, 1>& translation) {
    Eigen::Matrix<T, 3, 3> rotation_t = rotation.transpose();
    Eigen::Matrix<T, 3, 1> translation_i = -rotation_t * translation;
    return {rotation_t, translation_i};
}

template <typename T>
std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> product(
    const Eigen::Matrix<T, 3, 3>& rotation1, const Eigen::Matrix<T, 3, 1>& translation1,
    const Eigen::Matrix<T, 3, 3>& rotation2, const Eigen::Matrix<T, 3, 1>& translation2) {
    Eigen::Matrix<T, 3, 3> rotation = rotation1 * rotation2;
    Eigen::Matrix<T, 3, 1> translation = rotation1 * translation2 + translation1;
    return {rotation, translation};
}

inline void populate_quat_tran(const Eigen::Isometry3d& pose, std::array<double, 4>& quat,
                               std::array<double, 3>& translation) {
    Eigen::Quaterniond q0(pose.linear());
    quat = {q0.w(), q0.x(), q0.y(), q0.z()};
    translation = {pose.translation().x(), pose.translation().y(), pose.translation().z()};
}

inline Eigen::Quaterniond array_to_norm_quat(const std::array<double, 4>& arr) {
    Eigen::Quaterniond quat(arr[0], arr[1], arr[2], arr[3]);
    quat.normalize();
    return quat;
}

inline Eigen::Isometry3d restore_pose(const std::array<double, 4>& quat,
                                      const std::array<double, 3>& translation) {
    auto pose = Eigen::Isometry3d::Identity();
    pose.linear() = array_to_norm_quat(quat).toRotationMatrix();
    pose.translation() << translation[0], translation[1], translation[2];
    return pose;
}

// ---------- small SO(3) helpers (double) ----------
inline Eigen::Matrix3d projectToSO3(const Eigen::Matrix3d& rmtx) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(rmtx, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d umtx = svd.matrixU();
    Eigen::Matrix3d vmtx = svd.matrixV();
    Eigen::Matrix3d sigma = Eigen::Matrix3d::Identity();
    if ((umtx * vmtx.transpose()).determinant() < 0.0) sigma(2, 2) = -1.0;
    return umtx * sigma * vmtx.transpose();
}

// Utility: skew-symmetric matrix from vector
inline Eigen::Matrix3d skew(const Eigen::Vector3d& vec) {
    Eigen::Matrix3d skew_mtx;
    skew_mtx << 0, -vec.z(), vec.y(), vec.z(), 0, -vec.x(), -vec.y(), vec.x(), 0;
    return skew_mtx;
}

// log(R) as a 3-vector (axis*angle)
inline Eigen::Vector3d log_so3(const Eigen::Matrix3d& rot_in) {
    const Eigen::Matrix3d rotation = projectToSO3(rot_in);
    double cos_theta = (rotation.trace() - 1.0) * 0.5;
    cos_theta = std::min(1.0, std::max(-1.0, cos_theta));
    double theta = std::acos(cos_theta);
    if (theta < 1e-12) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d wvec;
    wvec << rotation(2, 1) - rotation(1, 2), rotation(0, 2) - rotation(2, 0),
        rotation(1, 0) - rotation(0, 1);
    wvec *= 0.5 / std::sin(theta);
    return wvec * theta;
}

inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d& wvec) {
    double theta = wvec.norm();
    if (theta < 1e-12) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d avec = wvec / theta;
    Eigen::Matrix3d askew = skew(avec);
    return Eigen::Matrix3d::Identity() + std::sin(theta) * askew +
           (1.0 - std::cos(theta)) * (askew * askew);
}

inline Eigen::VectorXd solve_llsq(const Eigen::MatrixXd& amtx, const Eigen::VectorXd& bvec) {
    return amtx.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bvec);
}

// ---------- stable ridge LS solve ----------
template <class Mat, class Vec>
Eigen::VectorXd ridge_llsq(const Mat& amtx, const Vec& bvec, double lambda = 1e-10) {
    const int ncols = static_cast<int>(amtx.cols());
    return (amtx.transpose() * amtx + lambda * Eigen::MatrixXd::Identity(ncols, ncols))
        .ldlt()
        .solve(amtx.transpose() * bvec);
}

inline Eigen::Vector3d log_rot(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd axisangle(R);
    return axisangle.axis() * axisangle.angle();
}

inline std::array<double, 6> pose_to_array(const Eigen::Isometry3d& pose) {
    Eigen::AngleAxisd axisangle(pose.linear());
    return {axisangle.axis().x() * axisangle.angle(),
            axisangle.axis().y() * axisangle.angle(),
            axisangle.axis().z() * axisangle.angle(),
            pose.translation().x(),
            pose.translation().y(),
            pose.translation().z()};
}

inline Eigen::Isometry3d array_to_pose(const double* pose) {
    Eigen::Matrix3d rotation;
    ceres::AngleAxisToRotationMatrix(pose, rotation.data());
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = rotation;
    T.translation() = Eigen::Vector3d(pose[3], pose[4], pose[5]);
    return T;
}

// Utility: average a set of affine transforms (rotation via quaternion averaging)
inline Eigen::Isometry3d average_affines(const std::vector<Eigen::Isometry3d>& poses) {
    if (poses.empty()) return Eigen::Isometry3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_sum(0, 0, 0, 0);
    for (const auto& p : poses) {
        translation += p.translation();
        Eigen::Quaterniond q(p.linear());
        if (q_sum.coeffs().dot(q.coeffs()) < 0.0) q.coeffs() *= -1.0;
        q_sum.coeffs() += q.coeffs();
    }
    translation /= static_cast<double>(poses.size());
    q_sum.normalize();
    Eigen::Isometry3d avg = Eigen::Isometry3d::Identity();
    avg.linear() = q_sum.toRotationMatrix();
    avg.translation() = translation;
    return avg;
}

template <typename T>
static void planar_observables_to_observables(
    const PlanarView& planar_obs, std::vector<Observation<T>>& obs,
    const Eigen::Transform<T, 3, Eigen::Isometry>& camera_se3_target) {
    if (obs.size() != planar_obs.size()) obs.resize(planar_obs.size());
    for (size_t i = 0; i < planar_obs.size(); ++i) {
        const auto& planar_item = planar_obs[i];
        // Convert pixel coordinates to normalized image coordinates
        Eigen::Matrix<T, 3, 1> point{T(planar_item.object_xy.x()), T(planar_item.object_xy.y()),
                                     T(0)};
        point = camera_se3_target * point;
        const T xn = point.x() / point.z();
        const T yn = point.y() / point.z();
        obs[i] = Observation<T>{xn, yn, T(planar_item.image_uv.x()), T(planar_item.image_uv.y())};
    }
}

template <typename T>
Observation<T> to_observation(const PlanarObservation& obs, const T* pose6) {
    const T* axisangle = pose6;        // angle-axis
    const T* translation = pose6 + 3;  // translation

    Eigen::Matrix<T, 3, 1> point{T(obs.object_xy.x()), T(obs.object_xy.y()), T(0.0)};
    Eigen::Matrix<T, 3, 1> point_c;
    ceres::AngleAxisRotatePoint(axisangle, point.data(), point_c.data());
    point_c += Eigen::Matrix<T, 3, 1>(translation[0], translation[1], translation[2]);
    T inv_z = T(1.0) / point_c.z();
    Observation<T> observation;
    observation.x = point_c.x() * inv_z;
    observation.y = point_c.y() * inv_z;
    observation.u = T(obs.image_uv.x());
    observation.v = T(obs.image_uv.y());
    return observation;
}

}  // namespace calib
