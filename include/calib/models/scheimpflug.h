#pragma once

// std
#include <cmath>

// eigen
#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>

#include "calib/models/camera_matrix.h"
#include "calib/models/cameramodel.h"

namespace calib {

struct ScheimpflugAngles final {
    double tau_x{0};  ///< Tilt around the X axis (radians)
    double tau_y{0};  ///< Tilt around the Y axis (radians)
};

/**
 * @brief Camera model with a tilted sensor plane (Scheimpflug configuration).
 *
 * The camera follows the central projection model but the image sensor is
 * tilted with respect to the optical axis. The tilt is parameterised by two
 * angles: tau_x is a rotation around the camera X axis and tau_y around the Y
 * axis. Distortion is applied in the metric coordinates on the tilted sensor
 * plane.
 *
 * @tparam CameraT Underlying camera model type that satisfies camera_model concept
 */
template <camera_model CameraT>
struct ScheimpflugCamera final {
    using Scalar = typename CameraT::Scalar;
    CameraT camera;   ///< Underlying camera model
    Scalar tau_x{0};  ///< Tilt around the X axis (radians)
    Scalar tau_y{0};  ///< Tilt around the Y axis (radians)

    /// @brief Default constructor - creates camera with zero tilt angles
    ScheimpflugCamera() : camera() {}

    /**
     * @brief Construct from camera and angle structure
     * @param cam Underlying camera model
     * @param angles Scheimpflug tilt angles (tau_x, tau_y)
     */
    ScheimpflugCamera(const CameraT& cam, ScheimpflugAngles angles)
        : camera(cam), tau_x(angles.tau_x), tau_y(angles.tau_y) {}

    /**
     * @brief Construct from different camera type and individual angles
     * @tparam OtherCamT Camera type (can differ from CameraT)
     * @tparam T Scalar type for angles
     * @param cam Underlying camera model
     * @param tau_x_angle Tilt around X axis (radians)
     * @param tau_y_angle Tilt around Y axis (radians)
     */
    template <camera_model OtherCamT, typename T>
    ScheimpflugCamera(OtherCamT cam, T tau_x_angle, T tau_y_angle)
        : camera(std::move(cam)), tau_x(Scalar(tau_x_angle)), tau_y(Scalar(tau_y_angle)) {}

    /// @brief Copy constructor
    ScheimpflugCamera(const ScheimpflugCamera& other)
        : camera(other.camera), tau_x(other.tau_x), tau_y(other.tau_y) {}

    /// @brief Copy assignment operator
    ScheimpflugCamera& operator=(const ScheimpflugCamera& other) {
        if (this != &other) {
            camera = other.camera;
            tau_x = other.tau_x;
            tau_y = other.tau_y;
        }
        return *this;
    }

    /// @brief Move constructor
    ScheimpflugCamera(ScheimpflugCamera&& other) noexcept
        : camera(std::move(other.camera)), tau_x(other.tau_x), tau_y(other.tau_y) {}

    /// @brief Move assignment operator
    ScheimpflugCamera& operator=(ScheimpflugCamera&& other) noexcept {
        if (this != &other) {
            camera = std::move(other.camera);
            tau_x = other.tau_x;
            tau_y = other.tau_y;
        }
        return *this;
    }

    /**
     * @brief Apply linear intrinsic parameters to plane coordinates
     *
     * Maps plane coordinates m = (mx, my) to pixel coordinates using linear
     * intrinsics only (no distortion). Delegates to underlying camera model.
     * Typically: apply_intrinsics([x,y]) = [fx*x + skew*y + cx, fy*y + cy]
     *
     * @tparam T Scalar type (double or ceres::Jet for automatic differentiation)
     * @param plane_point 2D coordinates on the sensor plane
     * @return 2D pixel coordinates after applying linear intrinsics
     */
    template <typename T>
    [[nodiscard]]
    Eigen::Matrix<T, 2, 1> apply_intrinsics(const Eigen::Matrix<T, 2, 1>& plane_point) const {
        return camera.template apply_intrinsics<T>(plane_point);
    }

    /**
     * @brief Remove linear intrinsic parameters from pixel coordinates
     *
     * Inverse operation: pixels -> plane coords m = (mx, my), linear part only.
     * Typically: remove_intrinsics([u,v]) = [(u-cx - skew*((v-cy)/fy))/fx, (v-cy)/fy]
     *
     * @tparam T Scalar type (double or ceres::Jet for automatic differentiation)
     * @param pixel 2D pixel coordinates
     * @return 2D plane coordinates after removing linear intrinsics
     */
    template <typename T>
    [[nodiscard]]
    Eigen::Matrix<T, 2, 1> remove_intrinsics(const Eigen::Matrix<T, 2, 1>& pixel) const {
        return camera.template remove_intrinsics<T>(pixel);
    }

    /**
     * @brief Project a 3D point in the camera frame to pixel coordinates.
     *
     * Performs central projection accounting for the tilted sensor plane.
     * The algorithm:
     * 1. Computes ray-plane intersection with the tilted sensor
     * 2. Finds the principal ray intersection point
     * 3. Applies distortion to the local delta coordinates
     * 4. Adds linear shift from principal intersection
     *
     * @tparam T Scalar type (double or ceres::Jet for automatic differentiation)
     * @param x_camera 3D point expressed in the camera frame
     * @return 2D pixel coordinates
     */
    template <typename T>
    [[nodiscard]]
    auto project(const Eigen::Matrix<T, 3, 1>& x_camera) const -> Eigen::Matrix<T, 2, 1> {
        // Build rotation that aligns the tilted sensor basis
        const T tau_x_val = T(tau_x);
        const T tau_y_val = T(tau_y);
        const T cos_tau_x = ceres::cos(tau_x_val);
        const T sin_tau_x = ceres::sin(tau_x_val);
        const T cos_tau_y = ceres::cos(tau_y_val);
        const T sin_tau_y = ceres::sin(tau_y_val);

        Eigen::Matrix<T, 3, 3> rot_sensor;
        rot_sensor << cos_tau_y, sin_tau_x * sin_tau_y, cos_tau_x * sin_tau_y, T(0), cos_tau_x,
            -sin_tau_x, -sin_tau_y, sin_tau_x * cos_tau_y, cos_tau_x * cos_tau_y;

        // Basis vectors of the tilted sensor plane
        Eigen::Matrix<T, 3, 1> axis_sensor = rot_sensor.col(0);
        Eigen::Matrix<T, 3, 1> base_sensor = rot_sensor.col(1);
        Eigen::Matrix<T, 3, 1> normal_sensor = rot_sensor.col(2);

        // Ray-plane intersection (d=1)
        T sden = normal_sensor.dot(x_camera);
        T mx = axis_sensor.dot(x_camera) / sden;
        T my = base_sensor.dot(x_camera) / sden;

        // Principal ray intersection with the plane
        const T s0 = normal_sensor.z();
        const T mx0 = axis_sensor.z() / s0;
        const T my0 = base_sensor.z() / s0;

        // Distort only local delta in plane coords (about principal intersection)
        Eigen::Matrix<T, 2, 1> dxy(mx - mx0, my - my0);

        // Distort+intrinsics of the local delta via the base model:
        Eigen::Matrix<T, 2, 1> px_delta =
            camera.template project<T>(Eigen::Matrix<T, 3, 1>(dxy.x(), dxy.y(), T(1)));

        // Linear shift from principal-ray intersection (no cx,cy inside):
        Eigen::Matrix<T, 2, 1> base_shift =
            CameraTraits<CameraT>::template apply_linear_intrinsics<T>(camera, {mx0, my0});

        return px_delta + base_shift;
    }

    /**
     * @brief Unproject pixel coordinates to sensor plane coordinates.
     *
     * Inverse of the projection operation. Computes coordinates on the tilted
     * sensor plane corresponding to given pixel coordinates. The algorithm:
     * 1. Removes principal ray offset using linear intrinsics
     * 2. Inverts distortion through underlying camera model
     * 3. Adds back the principal intersection coordinates
     *
     * @tparam T Scalar type (double or ceres::Jet for automatic differentiation)
     * @param pixel 2D pixel coordinates
     * @return 2D coordinates on the tilted sensor plane
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

        // Remove principal-offset *linear* shift via traits
        Eigen::Matrix<T, 2, 1> base_shift =
            CameraTraits<CameraT>::template apply_intrinsics_linear<T>(camera, {mx0, my0});

        // Invert base camera mapping for the delta (distortion + intrinsics)
        // Your base unproject returns the *canonical* [dx, dy] on z=1
        Eigen::Matrix<T, 2, 1> dxy = camera.template unproject<T>(pixel - base_shift);

        // Return plane coords around the actual plane origin
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
