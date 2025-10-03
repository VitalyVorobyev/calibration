#pragma once

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/io/serialization.h"

namespace calib {

// keep it simple aggregate for json io
template <typename Scalar>
struct CameraMatrixT final {
    Scalar fx = Scalar(0);
    Scalar fy = Scalar(0);
    Scalar cx = Scalar(0);
    Scalar cy = Scalar(0);
    Scalar skew = Scalar(0);
};

template <typename Scalar>
[[nodiscard]] auto matrix(const CameraMatrixT<Scalar>& cam) -> Eigen::Matrix<Scalar, 3, 3> {
    Eigen::Matrix<Scalar, 3, 3> kmtx = Eigen::Matrix<Scalar, 3, 3>::Zero();
    kmtx(0, 0) = cam.fx;
    kmtx(1, 1) = cam.fy;
    kmtx(0, 1) = cam.skew;
    kmtx(0, 2) = cam.cx;
    kmtx(1, 2) = cam.cy;
    kmtx(2, 2) = Scalar(1);
    return kmtx;
}

template <typename T, typename Scalar>
[[nodiscard]] auto normalize(const CameraMatrixT<Scalar>& cam, const Eigen::Matrix<T, 2, 1>& pixel)
    -> Eigen::Matrix<T, 2, 1> {
    T y_coord = (pixel.y() - T(cam.cy)) / T(cam.fy);
    T x_coord = (pixel.x() - T(cam.cx) - T(cam.skew) * y_coord) / T(cam.fx);
    return {x_coord, y_coord};
}

template <typename T, typename Scalar>
[[nodiscard]] auto denormalize(const CameraMatrixT<Scalar>& cam,
                               const Eigen::Matrix<T, 2, 1>& norm_xy) -> Eigen::Matrix<T, 2, 1> {
    return {T(cam.fx) * norm_xy.x() + T(cam.skew) * norm_xy.y() + T(cam.cx),
            T(cam.fy) * norm_xy.y() + T(cam.cy)};
}

using CameraMatrix = CameraMatrixT<double>;

struct CalibrationBounds final {
    static constexpr double k_fx_min = 0.0;
    static constexpr double k_fx_max = 2000.0;
    static constexpr double k_fy_min = 0.0;
    static constexpr double k_fy_max = 2000.0;
    static constexpr double k_cx_min = 0.0;
    static constexpr double k_cx_max = 1280.0;
    static constexpr double k_cy_min = 0.0;
    static constexpr double k_cy_max = 720.0;
    static constexpr double k_skew_min = -0.01;
    static constexpr double k_skew_max = 0.01;

    double fx_min = k_fx_min;
    double fx_max = k_fx_max;
    double fy_min = k_fy_min;
    double fy_max = k_fy_max;
    double cx_min = k_cx_min;
    double cx_max = k_cx_max;
    double cy_min = k_cy_min;
    double cy_max = k_cy_max;
    double skew_min = k_skew_min;
    double skew_max = k_skew_max;
};

}  // namespace calib
