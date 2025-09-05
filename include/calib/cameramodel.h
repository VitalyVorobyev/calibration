#pragma once

#include <Eigen/Core>
#include <concepts>

#include "calib/cameramatrix.h"

namespace calib {

template <typename Cam>
concept camera_model = requires(const Cam& cam,
                                Eigen::Matrix<typename Cam::Scalar, 3, 1> p3,
                                Eigen::Matrix<typename Cam::Scalar, 2, 1> p2) {
    typename Cam::Scalar;
    cam.kmtx.fx;
    cam.kmtx.fy;
    cam.kmtx.skew;
    {
        cam.template project<typename Cam::Scalar>(p3)
    } -> std::same_as<Eigen::Matrix<typename Cam::Scalar, 2, 1>>;
    {
        cam.template unproject<typename Cam::Scalar>(p2)
    } -> std::same_as<Eigen::Matrix<typename Cam::Scalar, 2, 1>>;
};

template <typename CamT>
struct CameraTraits;  // primary template

}  // namespace calib
