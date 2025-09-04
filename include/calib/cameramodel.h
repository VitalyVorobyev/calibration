#pragma once

#include <Eigen/Core>
#include <concepts>

namespace calib {

template <typename Cam>
concept camera_model = requires(const Cam& cam,
                                Eigen::Matrix<typename Cam::Scalar, 3, 1> point3d) {
    typename Cam::Scalar;
    { cam.template project<typename Cam::Scalar>(point3d) }
        -> std::same_as<Eigen::Matrix<typename Cam::Scalar, 2, 1>>;
};

template <typename Cam>
concept CameraModel = camera_model<Cam>;

template <typename CamT>
struct CameraTraits;  // primary template

}  // namespace calib
