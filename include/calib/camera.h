#pragma once

#include <Eigen/Core>
#include <array>

#include "calib/cameramatrix.h"
#include "calib/distortion.h"
#include "calib/cameramodel.h"

namespace calib {

template<distortion_model DistortionT>
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

// Camera traits specialisation for generic pinhole camera
template<distortion_model DistortionT>
struct CameraTraits<Camera<DistortionT>> {
    static constexpr size_t param_count = 10;
    static constexpr int idx_fx = 0;   ///< Index of focal length in x
    static constexpr int idx_fy = 1;   ///< Index of focal length in y
    static constexpr int idx_skew = 4; ///< Index of skew parameter

    template<typename T>
    static Camera<BrownConrady<T>> from_array(const T* intr) {
        CameraMatrixT<T> K{intr[0], intr[1], intr[2], intr[3], intr[4]};
        Eigen::Matrix<T, Eigen::Dynamic, 1> dist(5);
        dist << intr[5], intr[6], intr[7], intr[8], intr[9];
        return Camera<BrownConrady<T>>(K, dist);
    }

    static void to_array(const Camera<DistortionT>& cam,
                         std::array<double, param_count>& arr) {
        arr[0] = cam.K.fx; arr[1] = cam.K.fy;
        arr[2] = cam.K.cx; arr[3] = cam.K.cy;
        arr[4] = cam.K.skew;
        for (int i = 0; i < 5; ++i) arr[5 + i] = cam.distortion.coeffs[i];
    }
};

} // namespace calib

