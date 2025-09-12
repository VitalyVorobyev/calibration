#pragma once

// std
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

namespace calib {

template <typename Scalar>
struct CameraMatrixT final {
    Scalar fx, fy, cx, cy;
    Scalar skew = Scalar(0);

    Eigen::Matrix<Scalar, 3, 3> matrix() const {
        Eigen::Matrix<Scalar, 3, 3> K = Eigen::Matrix<Scalar, 3, 3>::Identity();
        K(0, 0) = fx;
        K(0, 1) = skew;
        K(0, 2) = cx;
        K(1, 1) = fy;
        K(1, 2) = cy;
        return K;
    }

    /**
     * @brief Normalizes a 2D pixel coordinate using the intrinsic camera parameters.
     *
     * This function transforms a 2D pixel coordinate into a normalized coordinate
     * system by subtracting the principal point (cx, cy) and dividing by the focal
     * lengths (fx, fy). The normalization is performed for each component of the
     * input pixel coordinate.
     *
     * @tparam T The scalar type of the input and output coordinates (e.g., float, double).
     * @param pix The 2D pixel coordinate to be normalized.
     * @return Eigen::Matrix<T, 2, 1> The normalized 2D coordinate.
     */
    template <typename T>
    [[nodiscard]]
    auto normalize(const Eigen::Matrix<T, 2, 1>& pixel) const -> Eigen::Matrix<T, 2, 1> {
        T y_coord = (pixel.y() - T(cy)) / T(fy);
        T x_coord = (pixel.x() - T(cx) - T(skew) * y_coord) / T(fx);
        return {x_coord, y_coord};
    }

    /**
     * @brief Denormalizes a 2D point using the camera's intrinsic parameters.
     *
     * This function applies the camera's intrinsic parameters (focal lengths and principal point
     * offsets) to transform a normalized 2D point into its corresponding pixel coordinates.
     *
     * @tparam T The scalar type of the input and output (e.g., float, double).
     * @param xy The normalized 2D point as an Eigen::Matrix<T, 2, 1>.
     * @return Eigen::Matrix<T, 2, 1> The denormalized 2D point in pixel coordinates.
     */
    template <typename T>
    [[nodiscard]]
    auto denormalize(const Eigen::Matrix<T, 2, 1>& norm_xy) const -> Eigen::Matrix<T, 2, 1> {
        return {T(fx) * norm_xy.x() + T(skew) * norm_xy.y() + T(cx), T(fy) * norm_xy.y() + T(cy)};
    }
};

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
