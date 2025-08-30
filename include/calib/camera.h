#pragma once

#include <Eigen/Core>

#include "calib/cameramatrix.h"
#include "calib/distortion.h"

namespace calib {

template<DistortionModel DistortionT>
class Camera final {
public:
    using Scalar = typename DistortionT::Scalar;
    CameraMatrixT<Scalar> K;      ///< Intrinsic camera matrix parameters
    DistortionT distortion;       ///< Distortion model

    Camera() = default;
    Camera(const CameraMatrixT<Scalar>& m, const DistortionT& d)
        : K(m), distortion(d) {}

    template<typename Derived>
    Camera(const CameraMatrixT<Scalar>& m, const Eigen::MatrixBase<Derived>& coeffs)
        : K(m), distortion(coeffs) {}

    template<typename T>
    Eigen::Matrix<T,2,1> normalize(const Eigen::Matrix<T,2,1>& pix) const {
        return K.template normalize<T>(pix);
    }

    template<typename T>
    Eigen::Matrix<T,2,1> denormalize(const Eigen::Matrix<T,2,1>& xy) const {
        return K.template denormalize<T>(xy);
    }

    template<typename T>
    Eigen::Matrix<T,2,1> distort(const Eigen::Matrix<T,2,1>& norm_xy) const {
        return distortion.template distort<T>(norm_xy);
    }

    template<typename T>
    Eigen::Matrix<T,2,1> undistort(const Eigen::Matrix<T,2,1>& dist_xy) const {
        return distortion.template undistort<T>(dist_xy);
    }

    template<typename T>
    Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,2,1>& norm_xy) const {
        return denormalize(distort(norm_xy));
    }

    template<typename T>
    Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,3,1>& xyz) const {
        Eigen::Matrix<T,2,1> norm_xy = xyz.hnormalized();
        return denormalize(distort(norm_xy));
    }

    template<typename T>
    Eigen::Matrix<T,2,1> unproject(const Eigen::Matrix<T,2,1>& pix) const {
        return undistort(normalize(pix));
    }
};

using PinholeCamera = Camera<DualDistortion>;

} // namespace calib

