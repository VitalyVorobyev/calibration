/** @brief Utility functions for camera calibration tests */

#pragma once

// std
#include <random>
#include <vector>
#include <cmath>
#include <numbers>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calib/pinhole.h"
#include "calib/bundle.h"
#include "calib/scheimpflug.h"

using calib::PinholeCamera;
using calib::BrownConradyd;
using calib::PlanarView;
using calib::BundleObservation;
using calib::ScheimpflugCamera;
using calib::distortion_model;

static inline double deg2rad(double d) { return d * std::numbers::pi / 180.0; }
static inline double rad2deg(double r) { return r * 180.0 / std::numbers::pi; }

inline double rotation_angle(const Eigen::Matrix3d& R) {
    double c = (R.trace() - 1.0) * 0.5;
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c); // [0,pi]
}

inline PlanarView make_view(const std::vector<Eigen::Vector2d>& obj,
                            const std::vector<Eigen::Vector2d>& img) {
    PlanarView view(obj.size());
    for (size_t i = 0; i < obj.size(); ++i) {
        view[i].object_xy = obj[i];
        view[i].image_uv = img[i];
    }
    return view;
}

inline Eigen::Isometry3d compute_camera_se3_target(
    const Eigen::Isometry3d& b_se3_t,
    const Eigen::Isometry3d& g_se3_c,
    const Eigen::Isometry3d& b_se3_g) {
    Eigen::Isometry3d c_se3_t = g_se3_c.inverse() * b_se3_g.inverse() * b_se3_t;
    return c_se3_t;
}

inline Eigen::Matrix3d axis_angle_to_R(const Eigen::Vector3d& axis, double angle){
    if (angle < 1e-16) return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

inline Eigen::Isometry3d make_pose(const Eigen::Vector3d& t, const Eigen::Vector3d& axis, double angle) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = axis_angle_to_R(axis, angle);
    T.translation() = t;
    return T;
}

/**
 * @brief Generates a sequence of 3D poses arranged in a circle with optional elevation and rotation.
 *
 * This function creates a vector of Eigen::Isometry3d transformations representing poses
 * distributed evenly along a circle in the XY-plane, with each pose optionally elevated along the Z-axis
 * and rotated around a specified axis.
 *
 * @param n        Number of poses to generate along the circle.
 * @param radius   Radius of the circle in the XY-plane.
 * @param z0       Initial Z-coordinate for the first pose.
 * @param z_step   Incremental step in Z for each subsequent pose.
 * @param rot_step Incremental rotation (in radians) applied to each pose.
 * @param axis_z   Z-component of the rotation axis (default is 1.0).
 * @return std::vector<Eigen::Isometry3d> Vector of generated poses as affine transformations.
 */
inline std::vector<Eigen::Isometry3d> make_circle_poses(int n, double radius, double z0,
                                                      double z_step, double rot_step,
                                                      double axis_z = 1.0) {
    std::vector<Eigen::Isometry3d> poses;
    poses.reserve(n);
    for (int i = 0; i < n; ++i) {
        double angle = i * 2.0 * std::numbers::pi / n;
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(radius * std::cos(angle),
                                          radius * std::sin(angle),
                                          z0 + z_step * i);
        Eigen::Vector3d axis(std::cos(angle), std::sin(angle), axis_z);
        T.linear() = Eigen::AngleAxisd(rot_step * i, axis.normalized()).toRotationMatrix();
        poses.push_back(T);
    }
    return poses;
}

/**
 * @brief Generates a set of BundleObservation objects by projecting a set of 2D object points
 *        onto multiple Scheimpflug cameras for various board poses.
 *
 * This function iterates over all provided board-to-global transformations and all cameras,
 * projecting the given object points into each camera's image plane using the provided
 * camera models and transformations. The resulting image points, along with the corresponding
 * object points, board pose, and camera index, are stored in BundleObservation objects.
 *
 * @tparam DistortionT The distortion model type used by the ScheimpflugCamera.
 * @param scs Vector of ScheimpflugCamera objects, each representing a camera with distortion.
 * @param g_se3_cs Vector of transformations from global to each camera coordinate system.
 * @param b_se3_t Transformation from board to target coordinate system.
 * @param obj Vector of 2D object points (e.g., calibration pattern points) in the target frame.
 * @param b_se3_gs Vector of transformations from board to global coordinate system for each pose.
 * @return std::vector<BundleObservation> Vector of observations, each containing the projected
 *         image points, the corresponding object points, the board pose, and the camera index.
 */
template <distortion_model DistortionT>
inline std::vector<BundleObservation> make_scheimpflug_observations(
    const std::vector<ScheimpflugCamera<PinholeCamera<DistortionT>>>& scs,
    const std::vector<Eigen::Isometry3d>& g_se3_cs,
    const Eigen::Isometry3d& b_se3_t,
    const std::vector<Eigen::Vector2d>& obj,
    const std::vector<Eigen::Isometry3d>& b_se3_gs) {
    std::vector<BundleObservation> obs;
    obs.reserve(b_se3_gs.size() * scs.size());
    for (const auto& btg : b_se3_gs) {
        for (size_t cam_idx = 0; cam_idx < scs.size(); ++cam_idx) {
            Eigen::Isometry3d c_se3_t = compute_camera_se3_target(b_se3_t, g_se3_cs[cam_idx], btg);
            std::vector<Eigen::Vector2d> img;
            img.reserve(obj.size());
            for (const auto& xy : obj) {
                Eigen::Vector3d P(xy.x(), xy.y(), 0);
                P = c_se3_t * P;
                img.push_back(scs[cam_idx].project(P));
            }
            obs.push_back({make_view(obj, img), btg, cam_idx});
        }
    }
    return obs;
}

template <distortion_model DistortionT>
inline std::vector<BundleObservation> make_bundle_observations(
    const std::vector<PinholeCamera<DistortionT>>& cams,
    const std::vector<Eigen::Isometry3d>& g_se3_cs,
    const Eigen::Isometry3d& b_se3_t,
    const std::vector<Eigen::Vector2d>& obj,
    const std::vector<Eigen::Isometry3d>& b_se3_gs) {
    std::vector<BundleObservation> obs;
    obs.reserve(b_se3_gs.size() * cams.size());
    for (const auto& btg : b_se3_gs) {
        for (size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx) {
            Eigen::Isometry3d c_se3_t = compute_camera_se3_target(b_se3_t, g_se3_cs[cam_idx], btg);
            std::vector<Eigen::Vector2d> img;
            img.reserve(obj.size());
            for (const auto& xy : obj) {
                Eigen::Vector3d P(xy.x(), xy.y(), 0);
                P = c_se3_t * P;
                img.push_back(cams[cam_idx].project(P));
            }
            obs.push_back({make_view(obj, img), btg, cam_idx});
        }
    }
    return obs;
}
