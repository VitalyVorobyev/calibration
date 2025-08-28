#pragma once

// std
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

namespace vitavision {

struct CameraMatrix final {
    double fx, fy, cx, cy;

    template<typename T>
    Eigen::Matrix<T,2,1> normalize(const Eigen::Matrix<T,2,1>& pix) const {
        return {
            (pix.x() - T(cx)) / T(fx),
            (pix.y() - T(cy)) / T(fy)
        };
    }

    template<typename T>
    Eigen::Matrix<T,2,1> denormalize(const Eigen::Matrix<T,2,1>& xy) const {
        return {
            T(fx) * xy.x() + T(cx),
            T(fy) * xy.y() + T(cy)
        };
    }
};

struct CalibrationBounds {
    double fx_min = 100.0;
    double fx_max = 2000.0;
    double fy_min = 100.0;
    double fy_max = 2000.0;
    double cx_min = 10.0;
    double cx_max = 1280.0;
    double cy_min = 10.0;
    double cy_max = 720.0;
};

} // namespace vitavision

