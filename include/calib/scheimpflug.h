#pragma once

// std
#include <cmath>

// eigen
#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>

#include "calib/camera.h"

namespace calib {

struct ScheimpflugAngles final {
    double tau_x{0};  ///< Tilt around the X axis (radians)
    double tau_y{0};  ///< Tilt around the Y axis (radians)
};

/**
 * @brief Camera model with a tilted sensor plane (Scheimpflug configuration).
 *
 * The camera follows the central projection model but the image sensor is
 * tilted with respect to the optical axis.  The tilt is parameterised by two
 * angles: tau_x is a rotation around the camera X axis and tau_y around the Y
 * axis.  Distortion is applied in the metric coordinates on the tilted sensor
 * plane.
 */
template <distortion_model DistortionT>
struct ScheimpflugCamera final {
    using Scalar = typename DistortionT::Scalar;
    Camera<DistortionT> camera;  ///< Intrinsics and distortion parameters
    Scalar tau_x{0};             ///< Tilt around the X axis (radians)
    Scalar tau_y{0};             ///< Tilt around the Y axis (radians)

    ScheimpflugCamera() = default;
    ScheimpflugCamera(const Camera<DistortionT>& cam, ScheimpflugAngles angles)
        : camera(cam), tau_x(angles.tau_x), tau_y(angles.tau_y) {}

    template <distortion_model OtherDistortionT, typename T>
    ScheimpflugCamera(const Camera<OtherDistortionT>& cam, T tau_x_angle, T tau_y_angle)
        : camera(Camera<DistortionT>(CameraMatrixT<Scalar>{Scalar(cam.K.fx), Scalar(cam.K.fy),
                                                           Scalar(cam.K.cx), Scalar(cam.K.cy)},
                                     cam.distortion.coeffs.template cast<Scalar>())),
          tau_x(Scalar(tau_x_angle)),
          tau_y(Scalar(tau_y_angle)) {}

    /**
     * @brief Project a 3D point in the camera frame to pixel coordinates.
     *
     * @tparam T Scalar type (double or ceres::Jet)
     * @param Xc 3D point expressed in the camera frame
     * @return Eigen::Matrix<T,2,1> Pixel coordinates
     */
    template <typename T>
    [[nodiscard]]
    auto project(const Eigen::Matrix<T, 3, 1>& X_camera) const -> Eigen::Matrix<T, 2, 1> {
        // Build rotation that aligns the tilted sensor basis
        const T tau_x_val = T(tau_x);
        const T tau_y_val = T(tau_y);
        const T cos_tau_x = ceres::cos(tau_x_val);
        const T sin_tau_x = ceres::sin(tau_x_val);
        const T cos_tau_y = ceres::cos(tau_y_val);
        const T sin_tau_y = ceres::sin(tau_y_val);

        Eigen::Matrix<T, 3, 3> rot_x;
        rot_x << T(1), T(0), T(0), T(0), cos_tau_x, -sin_tau_x, T(0), sin_tau_x, cos_tau_x;
        Eigen::Matrix<T, 3, 3> rot_y;
        rot_y << cos_tau_y, T(0), sin_tau_y, T(0), T(1), T(0), -sin_tau_y, T(0), cos_tau_y;
        Eigen::Matrix<T, 3, 3> rot_sensor = rot_y * rot_x;

        // Basis vectors of the tilted sensor plane
        Eigen::Matrix<T, 3, 1> axis_sensor = rot_sensor.col(0);
        Eigen::Matrix<T, 3, 1> base_sensor = rot_sensor.col(1);
        Eigen::Matrix<T, 3, 1> normal_sensor = rot_sensor.col(2);

        // Ray-plane intersection (d=1)
        T sden = normal_sensor.dot(X_camera);
        T mx = axis_sensor.dot(X_camera) / sden;
        T my = base_sensor.dot(X_camera) / sden;

        // Principal ray intersection with the plane
        const T s0 = normal_sensor.z();
        const T mx0 = axis_sensor.z() / s0;
        const T my0 = base_sensor.z() / s0;

        // Distortion in plane coordinates
        Eigen::Matrix<T, 2, 1> dxy(mx - mx0, my - my0);
        dxy = camera.distortion.distort(dxy);
        mx = dxy.x() + mx0;
        my = dxy.y() + my0;

        T u = T(camera.K.fx) * mx + T(camera.K.skew) * my + T(camera.K.cx);
        T v = T(camera.K.fy) * my + T(camera.K.cy);
        return {u, v};
    }
};

// Traits specialisation for Scheimpflug camera
template <distortion_model DistortionT>
struct CameraTraits<ScheimpflugCamera<DistortionT>> {
    static constexpr size_t param_count = 12;
    static constexpr int idx_fx = 0;    ///< Index of focal length in x
    static constexpr int idx_fy = 1;    ///< Index of focal length in y
    static constexpr int idx_skew = 4;  ///< Index of skew parameter

    static constexpr int k_num_dist_coeffs = 5;
    static constexpr int k_tau_x_idx = 5;
    static constexpr int k_tau_y_idx = 6;
    static constexpr int k_dist_start_idx = 7;
    template <typename T>
    static auto from_array(const T* intr) -> ScheimpflugCamera<BrownConrady<T>> {
        CameraMatrixT<T> k_matrix{intr[0], intr[1], intr[2], intr[3], intr[4]};
        Eigen::Matrix<T, Eigen::Dynamic, 1> dist(k_num_dist_coeffs);
        dist << intr[k_dist_start_idx], intr[k_dist_start_idx + 1], intr[k_dist_start_idx + 2],
            intr[k_dist_start_idx + 3], intr[k_dist_start_idx + 4];
        Camera<BrownConrady<T>> cam(k_matrix, dist);
        return ScheimpflugCamera<BrownConrady<T>>(cam, intr[k_tau_x_idx], intr[k_tau_y_idx]);
    }

    static void to_array(const ScheimpflugCamera<DistortionT>& cam,
                         std::array<double, param_count>& arr) {
        arr[0] = cam.camera.K.fx;
        arr[1] = cam.camera.K.fy;
        arr[2] = cam.camera.K.cx;
        arr[3] = cam.camera.K.cy;
        arr[4] = cam.camera.K.skew;
        arr[k_tau_x_idx] = cam.tau_x;
        arr[k_tau_y_idx] = cam.tau_y;
        for (int i = 0; i < k_num_dist_coeffs; ++i) {
            arr[k_dist_start_idx + i] = cam.camera.distortion.coeffs[i];
        }
    }
};

}  // namespace calib
