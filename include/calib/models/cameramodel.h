#pragma once

#include <Eigen/Core>
#include <concepts>

#include "calib/models/camera_matrix.h"

namespace calib {

/**
 * @brief Concept defining the interface requirements for camera model implementations.
 *
 * A camera model must provide methods for projecting 3D points to 2D image coordinates
 * and unprojecting 2D points to normalized camera coordinates. It must also support
 * applying and removing intrinsic camera parameters.
 *
 * @tparam Cam The camera model type to be validated
 *
 * Requirements:
 * - Must define a Scalar type (typically float or double)
 * - Must provide templated project() method: 3D point → 2D image coordinates
 * - Must provide templated unproject() method: 2D image coordinates → 2D normalized coordinates
 * - Must provide templated apply_intrinsics() method: normalized coordinates → image coordinates
 * - Must provide templated remove_intrinsics() method: image coordinates → normalized coordinates
 *
 * @see CameraMatrix for basic pinhole camera implementation
 * @see distortion_model concept for distortion handling requirements
 */
template <typename Cam>
concept camera_model = requires(const Cam& cam, Eigen::Matrix<typename Cam::Scalar, 3, 1> p3,
                                Eigen::Matrix<typename Cam::Scalar, 2, 1> p2) {
    typename Cam::Scalar;
    {
        cam.template project<typename Cam::Scalar>(p3)
    } -> std::same_as<Eigen::Matrix<typename Cam::Scalar, 2, 1>>;
    {
        cam.template unproject<typename Cam::Scalar>(p2)
    } -> std::same_as<Eigen::Matrix<typename Cam::Scalar, 2, 1>>;
    {
        cam.template apply_intrinsics<typename Cam::Scalar>(p2)
    } -> std::same_as<Eigen::Matrix<typename Cam::Scalar, 2, 1>>;
    {
        cam.template remove_intrinsics<typename Cam::Scalar>(p2)
    } -> std::same_as<Eigen::Matrix<typename Cam::Scalar, 2, 1>>;
};

/**
 * @brief Type traits template for camera model implementations.
 *
 * Provides compile-time information about camera model types. Specializations
 * should define camera-specific properties and metadata.
 *
 * @tparam CamT The camera model type
 *
 * Typical specializations should provide:
 * - Static constexpr properties describing the camera model
 * - Type aliases for scalar types and parameter representations
 * - Compile-time constants for parameter counts or model properties
 *
 * @note This is a primary template that should be specialized for each
 *       camera model implementation.
 */
template <typename CamT>
struct CameraTraits;  // primary template

}  // namespace calib
