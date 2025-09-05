/**
 * @file camera.h
 * @brief Pinhole camera model with intrinsics and distortion
 * @ingroup camera_calibration
 *
 * This file provides a unified pinhole camera model that combines intrinsic
 * parameters with lens distortion correction, supporting various distortion
 * models through the distortion_model concept.
 */

#pragma once

#include <Eigen/Core>
#include <array>

#include "calib/cameramatrix.h"
#include "calib/cameramodel.h"
#include "calib/distortion.h"

namespace calib {

/**
 * @brief Pinhole camera model with intrinsics and distortion correction
 * @ingroup camera_calibration
 *
 * This class represents a pinhole camera model combining intrinsic parameters
 * (focal length, principal point, skew) with lens distortion correction.  The
 * distortion model is templated and must satisfy the distortion_model concept.
 *
 * @tparam DistortionT Distortion model type (must satisfy distortion_model concept)
 *
 * Key features:
 * - Unified pixel-to-world and world-to-pixel transformations
 * - Template-based distortion model support
 * - Automatic type deduction for scalar types
 * - Efficient coordinate transformations
 */
template <distortion_model DistortionT>
class PinholeCamera final {
  public:
    using Scalar = typename DistortionT::Scalar;  ///< Scalar type from distortion model
    CameraMatrixT<Scalar> kmtx;                   ///< Intrinsic camera matrix parameters
    DistortionT distortion;                       ///< Distortion model instance

    /// Default constructor
    PinholeCamera() = default;

    /**
     * @brief Construct camera with intrinsic matrix and distortion model
     * @param matrix Intrinsic camera matrix
     * @param distortion_model Distortion model instance
     */
    PinholeCamera(const CameraMatrixT<Scalar>& matrix, const DistortionT& distortion_model)
        : kmtx(matrix), distortion(distortion_model) {}

    /**
     * @brief Construct camera with intrinsic matrix and distortion coefficients
     * @tparam Derived Eigen matrix expression type
     * @param matrix Intrinsic camera matrix
     * @param coeffs Distortion coefficients
     */
    template <typename Derived>
    PinholeCamera(const CameraMatrixT<Scalar>& matrix, const Eigen::MatrixBase<Derived>& coeffs)
        : kmtx(matrix), distortion(coeffs) {}

    /**
     * @brief Normalize pixel coordinates using intrinsic parameters
     * @tparam T Scalar type for computation
     * @param pixel 2D pixel coordinate
     * @return Normalized 2D coordinate
     */
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

// Pinhole camera traits specialisation
template <distortion_model DistortionT>
struct CameraTraits<PinholeCamera<DistortionT>> {
    static constexpr size_t param_count = 10;
    static constexpr int idx_fx = 0;    ///< Index of focal length in x
    static constexpr int idx_fy = 1;    ///< Index of focal length in y
    static constexpr int idx_skew = 4;  ///< Index of skew parameter

    static constexpr int k_num_dist_coeffs = 5;
    template <typename T>
    static auto from_array(const T* intr) -> PinholeCamera<BrownConrady<T>> {
        CameraMatrixT<T> k_matrix{intr[0], intr[1], intr[2], intr[3], intr[4]};
        Eigen::Matrix<T, Eigen::Dynamic, 1> dist(k_num_dist_coeffs);
        constexpr int k_intr_offset = 5;
        dist << intr[k_intr_offset], intr[k_intr_offset + 1], intr[k_intr_offset + 2],
            intr[k_intr_offset + 3], intr[k_intr_offset + 4];
        return PinholeCamera<BrownConrady<T>>(k_matrix, dist);
    }

    static void to_array(const PinholeCamera<DistortionT>& cam,
                         std::array<double, param_count>& arr) {
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

// Backwards compatibility alias
template <distortion_model DistortionT>
using Camera = PinholeCamera<DistortionT>;

}  // namespace calib
