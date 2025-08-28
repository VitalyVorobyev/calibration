#pragma once

// eigen
#include <Eigen/Core>

#include "calibration/cameramatrix.h"
#include "calibration/distortion.h"

namespace vitavision {

class Camera final {
public:
    CameraMatrix K;           ///< Intrinsic camera matrix parameters
    DualDistortion distortion; ///< Forward and inverse distortion coefficients

    Camera() = default;
    Camera(const CameraMatrix& m, const DualDistortion& d)
        : K(m), distortion(d) {}

    template<typename T>
    Eigen::Matrix<T,2,1> normalize(const Eigen::Matrix<T,2,1>& pix) const {
        return K.normalize(pix);
    }

    template<typename T>
    Eigen::Matrix<T,2,1> denormalize(const Eigen::Matrix<T,2,1>& xy) const {
        return K.denormalize(xy);
    }

    template<typename T>
    Eigen::Matrix<T,2,1> distort(const Eigen::Matrix<T,2,1>& norm_xy) const {
        return distortion.distort(norm_xy);
    }

    template<typename T>
    Eigen::Matrix<T,2,1> undistort(const Eigen::Matrix<T,2,1>& dist_xy) const {
        return distortion.undistort(dist_xy);
    }

    template<typename T>
    Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,2,1>& norm_xy) const {
        return denormalize(distort(norm_xy));
    }

    template<typename T>
    Eigen::Matrix<T,2,1> unproject(const Eigen::Matrix<T,2,1>& pix) const {
        return undistort(normalize(pix));
    }
};

} // namespace vitavision

