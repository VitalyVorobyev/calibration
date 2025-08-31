#pragma once

#include <Eigen/Core>
#include <concepts>
#include <array>

namespace calib {

template<typename Cam>
concept camera_model = requires(const Cam& cam, Eigen::Matrix<typename Cam::Scalar,3,1> P) {
    typename Cam::Scalar;
    { cam.template project<typename Cam::Scalar>(P) } -> std::same_as<Eigen::Matrix<typename Cam::Scalar,2,1>>;
};

template<typename CamT>
struct CameraTraits; // primary template

} // namespace calib
