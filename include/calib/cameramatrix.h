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
    double fx_min = 0;
    double fx_max = 2000.0;
    double fy_min = 0;
    double fy_max = 2000.0;
    double cx_min = 0;
    double cx_max = 1280.0;
    double cy_min = 0;
    double cy_max = 720.0;
    double skew_min = -0.01;
    double skew_max = 0.01;
};

}  // namespace calib
