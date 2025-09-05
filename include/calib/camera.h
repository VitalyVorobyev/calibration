#pragma once

#include <Eigen/Core>
#include <array>

#include "calib/cameramatrix.h"
#include "calib/cameramodel.h"
#include "calib/distortion.h"

namespace calib {

template <distortion_model DistortionT>
class Camera final {
  public:
    using Scalar = typename DistortionT::Scalar;
    CameraMatrixT<Scalar> kmtx;  ///< Intrinsic camera matrix parameters
    DistortionT distortion;      ///< Distortion model

    Camera() = default;
    Camera(const CameraMatrixT<Scalar>& matrix, const DistortionT& distortion_model)
        : kmtx(matrix), distortion(distortion_model) {}

    template <typename Derived>
    Camera(const CameraMatrixT<Scalar>& matrix, const Eigen::MatrixBase<Derived>& coeffs)
        : kmtx(matrix), distortion(coeffs) {}

    template <typename T>
    [[nodiscard]]
    auto normalize(const Eigen::Matrix<T, 2, 1>& pixel) const -> Eigen::Matrix<T, 2, 1> {
        return kmtx.template normalize<T>(pixel);
    }

    template <typename T>
    [[nodiscard]]
    auto denormalize(const Eigen::Matrix<T, 2, 1>& norm_xy) const -> Eigen::Matrix<T, 2, 1> {
        return kmtx.template denormalize<T>(norm_xy);
    }

    template <typename T>
    [[nodiscard]]
    auto distort(const Eigen::Matrix<T, 2, 1>& norm_xy) const -> Eigen::Matrix<T, 2, 1> {
        return distortion.template distort<T>(norm_xy);
    }

    template <typename T>
    [[nodiscard]]
    auto undistort(const Eigen::Matrix<T, 2, 1>& distorted_xy) const -> Eigen::Matrix<T, 2, 1> {
        return distortion.template undistort<T>(distorted_xy);
    }

    template <typename T>
    [[nodiscard]]
    auto project(const Eigen::Matrix<T, 2, 1>& norm_xy) const -> Eigen::Matrix<T, 2, 1> {
        return denormalize(distort(norm_xy));
    }

    template <typename T>
    [[nodiscard]]
    auto project(const Eigen::Matrix<T, 3, 1>& xyz) const -> Eigen::Matrix<T, 2, 1> {
        Eigen::Matrix<T, 2, 1> norm_xy = xyz.hnormalized();
        return denormalize(distort(norm_xy));
    }

    template <typename T>
    [[nodiscard]]
    auto unproject(const Eigen::Matrix<T, 2, 1>& pixel) const -> Eigen::Matrix<T, 2, 1> {
        return undistort(normalize(pixel));
    }
};

// Camera traits specialisation for generic pinhole camera
template <distortion_model DistortionT>
struct CameraTraits<Camera<DistortionT>> {
    static constexpr size_t param_count = 10;
    static constexpr int idx_fx = 0;    ///< Index of focal length in x
    static constexpr int idx_fy = 1;    ///< Index of focal length in y
    static constexpr int idx_skew = 4;  ///< Index of skew parameter

    static constexpr int k_num_dist_coeffs = 5;
    template <typename T>
    static auto from_array(const T* intr) -> Camera<BrownConrady<T>> {
        CameraMatrixT<T> k_matrix{intr[0], intr[1], intr[2], intr[3], intr[4]};
        Eigen::Matrix<T, Eigen::Dynamic, 1> dist(k_num_dist_coeffs);
        constexpr int k_intr_offset = 5;
        dist << intr[k_intr_offset], intr[k_intr_offset + 1], intr[k_intr_offset + 2],
            intr[k_intr_offset + 3], intr[k_intr_offset + 4];
        return Camera<BrownConrady<T>>(k_matrix, dist);
    }

    static void to_array(const Camera<DistortionT>& cam, std::array<double, param_count>& arr) {
        arr[0] = cam.kmtx.fx;
        arr[1] = cam.kmtx.fy;
        arr[2] = cam.kmtx.cx;
        arr[3] = cam.kmtx.cy;
        arr[4] = cam.kmtx.skew;
        constexpr int k_intr_offset = 5;
        for (int i = 0; i < k_num_dist_coeffs; ++i) {
            arr[k_intr_offset + i] = cam.distortion.coeffs[i];
        }
    }
};

}  // namespace calib
