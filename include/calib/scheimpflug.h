#pragma once

// std
#include <cmath>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

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
template<distortion_model DistortionT>
struct ScheimpflugCamera final {
    using Scalar = typename DistortionT::Scalar;
    Camera<DistortionT> camera;   ///< Intrinsics and distortion parameters
    Scalar tau_x{0}; ///< Tilt around the X axis (radians)
    Scalar tau_y{0}; ///< Tilt around the Y axis (radians)

    ScheimpflugCamera() = default;
    ScheimpflugCamera(const Camera<DistortionT>& cam, Scalar tx, Scalar ty)
        : camera(cam), tau_x(tx), tau_y(ty) {}

    template<typename T>
    ScheimpflugCamera(const Camera<DistortionT>& cam, T tx, T ty)
        : camera(cam), tau_x(Scalar(tx)), tau_y(Scalar(ty)) {}

    template<distortion_model OtherDistortionT, typename T>
    ScheimpflugCamera(const Camera<OtherDistortionT>& cam, T tx, T ty)
        : camera(Camera<DistortionT>(CameraMatrixT<Scalar>{Scalar(cam.K.fx), Scalar(cam.K.fy), Scalar(cam.K.cx), Scalar(cam.K.cy)}, 
                                   cam.distortion.coeffs.template cast<Scalar>())), 
          tau_x(Scalar(tx)), tau_y(Scalar(ty)) {}

    /**
     * @brief Project a 3D point in the camera frame to pixel coordinates.
     *
     * @tparam T Scalar type (double or ceres::Jet)
     * @param Xc 3D point expressed in the camera frame
     * @return Eigen::Matrix<T,2,1> Pixel coordinates
     */
    template <typename T>
    Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,3,1>& Xc) const {
        // Build rotation that aligns the tilted sensor basis
        const T tx = T(tau_x);
        const T ty = T(tau_y);
        const T ctx = ceres::cos(tx);
        const T stx = ceres::sin(tx);
        const T cty = ceres::cos(ty);
        const T sty = ceres::sin(ty);

        Eigen::Matrix<T,3,3> Rx;
        Rx << T(1), T(0), T(0),
              T(0), ctx, -stx,
              T(0), stx,  ctx;
        Eigen::Matrix<T,3,3> Ry;
        Ry << cty, T(0), sty,
              T(0), T(1), T(0),
              -sty, T(0), cty;
        Eigen::Matrix<T,3,3> Rs = Ry * Rx;

        // Basis vectors of the tilted sensor plane
        Eigen::Matrix<T,3,1> as = Rs.col(0);
        Eigen::Matrix<T,3,1> bs = Rs.col(1);
        Eigen::Matrix<T,3,1> ns = Rs.col(2);

        // Ray-plane intersection (d=1)
        T sden = ns.dot(Xc);
        T mx = as.dot(Xc) / sden;
        T my = bs.dot(Xc) / sden;

        // Principal ray intersection with the plane
        const T s0 = ns.z();
        const T mx0 = as.z() / s0;
        const T my0 = bs.z() / s0;

        // Distortion in plane coordinates
        Eigen::Matrix<T,2,1> dxy(mx - mx0, my - my0);
        dxy = camera.distortion.distort(dxy);
        mx = dxy.x() + mx0;
        my = dxy.y() + my0;

        // Pixel mapping (no skew)
        T u = T(camera.K.fx) * mx + T(camera.K.cx);
        T v = T(camera.K.fy) * my + T(camera.K.cy);
        return {u, v};
    }
};

// Traits specialisation for Scheimpflug camera
template<distortion_model DistortionT>
struct CameraTraits<ScheimpflugCamera<DistortionT>> {
    static constexpr size_t param_count = 11;

    template<typename T>
    static ScheimpflugCamera<BrownConrady<T>> from_array(const T* intr) {
        CameraMatrixT<T> K{intr[0], intr[1], intr[2], intr[3]};
        Eigen::Matrix<T, Eigen::Dynamic, 1> dist(5);
        dist << intr[6], intr[7], intr[8], intr[9], intr[10];
        Camera<BrownConrady<T>> cam(K, dist);
        return ScheimpflugCamera<BrownConrady<T>>(cam, intr[4], intr[5]);
    }

    static void to_array(const ScheimpflugCamera<DistortionT>& cam,
                         std::array<double, param_count>& arr) {
        arr[0] = cam.camera.K.fx; arr[1] = cam.camera.K.fy;
        arr[2] = cam.camera.K.cx; arr[3] = cam.camera.K.cy;
        arr[4] = cam.tau_x; arr[5] = cam.tau_y;
        for (int i = 0; i < 5; ++i) arr[6 + i] = cam.camera.distortion.coeffs[i];
    }
};

} // namespace calib
