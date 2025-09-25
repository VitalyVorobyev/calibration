/** @brief Utility functions for working with image observations */

#pragma once

// ceres
#include "../common/se3_utils.h"
#include "calib/estimation/planarpose.h"
#include "calib/models/distortion.h"
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
// Utility: skew-symmetric matrix from vector
// log(R) as a 3-vector (axis*angle)
// ---------- stable ridge LS solve ----------
inline Eigen::Isometry3d array_to_pose(const double* pose) {
    Eigen::Matrix3d rotation;
    ceres::AngleAxisToRotationMatrix(pose, rotation.data());
    Eigen::Isometry3d se3 = Eigen::Isometry3d::Identity();
    se3.linear() = rotation;
    se3.translation() = Eigen::Vector3d(pose[3], pose[4], pose[5]);
    return se3;
}

// Utility: average a set of affine transforms (rotation via quaternion averaging)
template <typename T>
static void planar_observables_to_observables(
    const PlanarView& planar_obs, std::vector<Observation<T>>& obs,
    const Eigen::Transform<T, 3, Eigen::Isometry>& camera_se3_target) {
    if (obs.size() != planar_obs.size()) {
        obs.resize(planar_obs.size());
    }
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
