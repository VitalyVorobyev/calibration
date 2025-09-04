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
    ScheimpflugCamera(const Camera<DistortionT>& cam, Scalar tau_x_angle, Scalar tau_y_angle)
        : camera(cam), tau_x(tau_x_angle), tau_y(tau_y_angle) {}

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

    /**
     * @brief Unproject pixel to a camera-frame ray with z=1 convention.
     *        Inverts intrinsics (with skew), undistorts in tilted-plane local coords,
     *        then maps back through the tilted sensor basis.
     *
     * @tparam T Scalar type (double or ceres::Jet)
     * @param px Pixel coordinates (u, v)
     * @return Eigen::Matrix<T,3,1> Ray direction in camera frame with z=1
     */
    template <typename T>
    [[nodiscard]]
    auto unproject(const Eigen::Matrix<T, 2, 1>& px) const -> Eigen::Matrix<T, 2, 1> {
        // 1) Build the same tilt rotation used in project()
        const T tx = T(tau_x);
        const T ty = T(tau_y);
        const T cx = ceres::cos(tx), sx = ceres::sin(tx);
        const T cy = ceres::cos(ty), sy = ceres::sin(ty);

        Eigen::Matrix<T, 3, 3> Rx;  // tilt about X
        Rx << T(1), T(0), T(0),
              T(0),   cx,  -sx,
              T(0),   sx,    cx;

        Eigen::Matrix<T, 3, 3> Ry;  // tilt about Y
        Ry <<   cy, T(0),   sy,
               T(0), T(1), T(0),
               -sy, T(0),   cy;

        const Eigen::Matrix<T,3,3> R = Ry * Rx;
        const Eigen::Matrix<T,3,1> e1 = R.col(0); // axis_sensor
        const Eigen::Matrix<T,3,1> e2 = R.col(1); // base_sensor
        const Eigen::Matrix<T,3,1> n  = R.col(2); // normal_sensor

        // 2) Principal-ray intersection offsets on the tilted plane (d=1)
        //    These are exactly the mx0,my0 used in project().
        const T s0  = n.z();                      // n·e_z
        const T mx0 = e1.z() / s0;
        const T my0 = e2.z() / s0;

        // 3) Invert intrinsics (with skew) to get distorted plane coords (mx,my)
        const T fx = T(camera.K.fx);
        const T fy = T(camera.K.fy);
        const T cxp = T(camera.K.cx);
        const T cyp = T(camera.K.cy);
        const T skew = T(camera.K.skew);

        const T u = px.x();
        const T v = px.y();

        T my = (v - cyp) / fy;
        T mx = (u - cxp - skew * my) / fx;

        // 4) Remove principal offset and undistort in local plane coords
        Eigen::Matrix<T,2,1> dxy_d(mx - mx0, my - my0);     // distorted delta about principal intersection
        Eigen::Matrix<T,2,1> dxy_u = camera.distortion.undistort(dxy_d);
        mx = dxy_u.x() + mx0;
        my = dxy_u.y() + my0;

        // 5) Convert (mx,my) on the tilted plane back to a camera-frame ray.
        //    In the orthonormal basis {e1,e2,n}, a direction with plane coords (mx,my) is:
        //      r ~ mx*e1 + my*e2 + 1*n  = R * [mx, my, 1]^T
        Eigen::Matrix<T,3,1> r = R * (Eigen::Matrix<T,3,1>() << mx, my, T(1)).finished();

        // 6) Normalize to z=1 (like pinhole), so downstream code can assume "normalized ray".
        const T invz = T(1) / r.z();
        return { r.x() * invz, r.y() * invz };
    }
};

// Traits specialisation for Scheimpflug camera
template <distortion_model DistortionT>
struct CameraTraits<ScheimpflugCamera<DistortionT>> {
    static constexpr size_t param_count = 12;
    static constexpr int idx_fx = 0;    ///< Index of focal length in x
    static constexpr int idx_fy = 1;    ///< Index of focal length in y
    static constexpr int idx_skew = 4;  ///< Index of skew parameter

    static constexpr int kNumDistCoeffs = 5;
    static constexpr int kTauXIdx = 5;
    static constexpr int kTauYIdx = 6;
    static constexpr int kDistStartIdx = 7;
    template <typename T>
    static auto from_array(const T* intr) -> ScheimpflugCamera<BrownConrady<T>> {
        CameraMatrixT<T> k_matrix{intr[0], intr[1], intr[2], intr[3], intr[4]};
        Eigen::Matrix<T, Eigen::Dynamic, 1> dist(kNumDistCoeffs);
        dist << intr[kDistStartIdx], intr[kDistStartIdx + 1], intr[kDistStartIdx + 2],
            intr[kDistStartIdx + 3], intr[kDistStartIdx + 4];
        Camera<BrownConrady<T>> cam(k_matrix, dist);
        return ScheimpflugCamera<BrownConrady<T>>(cam, intr[kTauXIdx], intr[kTauYIdx]);
    }

    static void to_array(const ScheimpflugCamera<DistortionT>& cam,
                         std::array<double, param_count>& arr) {
        arr[0] = cam.camera.K.fx;
        arr[1] = cam.camera.K.fy;
        arr[2] = cam.camera.K.cx;
        arr[3] = cam.camera.K.cy;
        arr[4] = cam.camera.K.skew;
        arr[kTauXIdx] = cam.tau_x;
        arr[kTauYIdx] = cam.tau_y;
        for (int i = 0; i < kNumDistCoeffs; ++i) {
            arr[kDistStartIdx + i] = cam.camera.distortion.coeffs[i];
        }
    }
};

}  // namespace calib
