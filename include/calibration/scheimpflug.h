#pragma once

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include "calibration/camera.h"

namespace vitavision {

/**
 * @brief Camera model with a tilted sensor plane (Scheimpflug configuration).
 *
 * The camera follows the central projection model but the image sensor is
 * tilted with respect to the optical axis.  The tilt is parameterised by two
 * angles: tau_x is a rotation around the camera X axis and tau_y around the Y
 * axis.  Distortion is applied in the metric coordinates on the tilted sensor
 * plane.
 */
struct ScheimpflugCamera final {
    Camera camera;   ///< Intrinsics and distortion parameters
    double tau_x{0}; ///< Tilt around the X axis (radians)
    double tau_y{0}; ///< Tilt around the Y axis (radians)

    ScheimpflugCamera() = default;
    ScheimpflugCamera(const Camera& cam, double tx, double ty)
        : camera(cam), tau_x(tx), tau_y(ty) {}

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
        const T ctx = std::cos(tx);
        const T stx = std::sin(tx);
        const T cty = std::cos(ty);
        const T sty = std::sin(ty);

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
        T s0 = ns.z();
        T mx0 = as.z() / s0;
        T my0 = bs.z() / s0;

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

} // namespace vitavision

