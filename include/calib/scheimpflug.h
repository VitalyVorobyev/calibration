#pragma once

// std
#include <cmath>

// eigen
#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>

#include "calib/cameramodel.h"
#include "calib/cameramatrix.h"

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
template <camera_model CameraT>
struct ScheimpflugCamera final {
    using Scalar = typename CameraT::Scalar;
    CameraT camera;             ///< Underlying camera model
    CameraMatrixT<Scalar>& kmtx;///< Reference to intrinsic matrix
    Scalar tau_x{0};            ///< Tilt around the X axis (radians)
    Scalar tau_y{0};            ///< Tilt around the Y axis (radians)

    ScheimpflugCamera() : camera(), kmtx(camera.kmtx) {}
    ScheimpflugCamera(const CameraT& cam, ScheimpflugAngles angles)
        : camera(cam), kmtx(camera.kmtx), tau_x(angles.tau_x), tau_y(angles.tau_y) {}

    template <camera_model OtherCamT, typename T>
    ScheimpflugCamera(const OtherCamT& cam, T tau_x_angle, T tau_y_angle)
        : camera(cam), kmtx(camera.kmtx), tau_x(Scalar(tau_x_angle)), tau_y(Scalar(tau_y_angle)) {}

    ScheimpflugCamera(const ScheimpflugCamera& other)
        : camera(other.camera), kmtx(camera.kmtx), tau_x(other.tau_x), tau_y(other.tau_y) {}

    ScheimpflugCamera& operator=(const ScheimpflugCamera& other) {
        if (this != &other) {
            camera = other.camera;
            tau_x = other.tau_x;
            tau_y = other.tau_y;
        }
        return *this;
    }

    ScheimpflugCamera(ScheimpflugCamera&& other) noexcept
        : camera(std::move(other.camera)), kmtx(camera.kmtx),
          tau_x(other.tau_x), tau_y(other.tau_y) {}

    ScheimpflugCamera& operator=(ScheimpflugCamera&& other) noexcept {
        if (this != &other) {
            camera = std::move(other.camera);
            tau_x = other.tau_x;
            tau_y = other.tau_y;
        }
        return *this;
    }

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

        // Normalised coordinates relative to principal ray
        Eigen::Matrix<T, 2, 1> dxy(mx - mx0, my - my0);

        // Base pixel shift due to principal ray offset
        Eigen::Matrix<T, 2, 1> base_shift(T(camera.kmtx.fx) * mx0 + T(camera.kmtx.skew) * my0,
                                          T(camera.kmtx.fy) * my0);

        // Delegate distortion and intrinsic mapping to the internal camera
        Eigen::Matrix<T, 2, 1> px =
            camera.template project<T>(Eigen::Matrix<T, 3, 1>(dxy.x(), dxy.y(), T(1)));

        return px + base_shift;
    }

    /**
     * @brief Unproject pixel coordinates to sensor plane coordinates.
     *
     * @tparam T Scalar type (double or ceres::Jet)
     * @param pixel Pixel coordinates
     * @return Eigen::Matrix<T,2,1> Coordinates on the tilted sensor plane
     */
    template <typename T>
    [[nodiscard]]
    auto unproject(const Eigen::Matrix<T, 2, 1>& pixel) const -> Eigen::Matrix<T, 2, 1> {
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

        Eigen::Matrix<T, 3, 1> axis_sensor = rot_sensor.col(0);
        Eigen::Matrix<T, 3, 1> base_sensor = rot_sensor.col(1);
        Eigen::Matrix<T, 3, 1> normal_sensor = rot_sensor.col(2);

        const T s0 = normal_sensor.z();
        const T mx0 = axis_sensor.z() / s0;
        const T my0 = base_sensor.z() / s0;

        Eigen::Matrix<T, 2, 1> base_shift(T(camera.kmtx.fx) * mx0 + T(camera.kmtx.skew) * my0,
                                          T(camera.kmtx.fy) * my0);

        Eigen::Matrix<T, 2, 1> dxy =
            camera.template unproject<T>(pixel - base_shift);

        return {dxy.x() + mx0, dxy.y() + my0};
    }
};

// Traits specialisation for Scheimpflug camera
template <camera_model CameraT>
struct CameraTraits<ScheimpflugCamera<CameraT>> {
    static constexpr size_t param_count = CameraTraits<CameraT>::param_count + 2;
    static constexpr int idx_fx = CameraTraits<CameraT>::idx_fx;
    static constexpr int idx_fy = CameraTraits<CameraT>::idx_fy;
    static constexpr int idx_skew = CameraTraits<CameraT>::idx_skew;

    static constexpr int k_tau_x_idx = CameraTraits<CameraT>::param_count;
    static constexpr int k_tau_y_idx = CameraTraits<CameraT>::param_count + 1;

    template <typename T>
    static auto from_array(const T* intr)
        -> ScheimpflugCamera<decltype(CameraTraits<CameraT>::from_array(intr))> {
        auto cam = CameraTraits<CameraT>::from_array(intr);
        return ScheimpflugCamera<decltype(cam)>(cam, intr[k_tau_x_idx], intr[k_tau_y_idx]);
    }

    static void to_array(const ScheimpflugCamera<CameraT>& cam,
                         std::array<double, param_count>& arr) {
        std::array<double, CameraTraits<CameraT>::param_count> inner{};
        CameraTraits<CameraT>::to_array(cam.camera, inner);
        for (size_t i = 0; i < CameraTraits<CameraT>::param_count; ++i) {
            arr[i] = inner[i];
        }
        arr[k_tau_x_idx] = cam.tau_x;
        arr[k_tau_y_idx] = cam.tau_y;
    }
};

}  // namespace calib
