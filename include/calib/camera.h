#pragma once

// eigen
#include <Eigen/Core>

#include "calib/cameramatrix.h"
#include "calib/distortion.h"

namespace calib {

class Camera final {
public:
    CameraMatrix K;            ///< Intrinsic camera matrix parameters
    DualDistortion distortion; ///< Forward and inverse distortion coefficients

    Camera() = default;
    Camera(const CameraMatrix& m, const DualDistortion& d)
        : K(m), distortion(d) {}
    Camera(const CameraMatrix& m, const Eigen::VectorXd& d)
        : K(m), distortion(d) {}

    /**
     * @brief Normalizes a 2D pixel coordinate using the camera's intrinsic parameters.
     *
     * This function takes a 2D pixel coordinate and applies the camera's intrinsic
     * normalization to transform it into a normalized coordinate system.
     *
     * @tparam T The scalar type of the matrix elements (e.g., float, double).
     * @param pix The 2D pixel coordinate to be normalized.
     * @return Eigen::Matrix<T,2,1> The normalized 2D coordinate.
     */
    template<typename T>
    Eigen::Matrix<T,2,1> normalize(const Eigen::Matrix<T,2,1>& pix) const {
        return K.normalize(pix);
    }

    /**
     * @brief Denormalizes a 2D point using the camera's intrinsic matrix.
     *
     * This function takes a 2D point in normalized coordinates and converts it
     * to pixel coordinates using the intrinsic matrix of the camera.
     *
     * @tparam T The scalar type of the input and output matrix (e.g., float, double).
     * @param xy The 2D point in normalized coordinates.
     * @return Eigen::Matrix<T,2,1> The 2D point in pixel coordinates.
     */
    template<typename T>
    Eigen::Matrix<T,2,1> denormalize(const Eigen::Matrix<T,2,1>& xy) const {
        return K.denormalize(xy);
    }

    /**
     * @brief Applies distortion to a normalized 2D point.
     *
     * This function takes a normalized 2D point (norm_xy) and applies the
     * distortion model to compute the distorted 2D point.
     *
     * @tparam T The scalar type of the input and output points (e.g., float, double).
     * @param norm_xy The normalized 2D point represented as an Eigen::Matrix<T, 2, 1>.
     * @return Eigen::Matrix<T, 2, 1> The distorted 2D point.
     */
    template<typename T>
    Eigen::Matrix<T,2,1> distort(const Eigen::Matrix<T,2,1>& norm_xy) const {
        return distortion.distort(norm_xy);
    }

    /**
     * @brief Undistorts a given 2D point using the camera's distortion model.
     *
     * This function takes a distorted 2D point and applies the camera's distortion
     * correction model to compute the undistorted point.
     *
     * @tparam T The scalar type of the input and output points (e.g., float, double).
     * @param dist_xy The distorted 2D point represented as an Eigen::Matrix<T, 2, 1>.
     * @return Eigen::Matrix<T, 2, 1> The undistorted 2D point.
     */
    template<typename T>
    Eigen::Matrix<T,2,1> undistort(const Eigen::Matrix<T,2,1>& dist_xy) const {
        return distortion.undistort(dist_xy);
    }

    /**
     * @brief Projects a normalized 2D point to pixel coordinates.
     *
     * This function takes a normalized 2D point (norm_xy), applies the distortion model,
     * and then converts it to pixel coordinates using the camera's intrinsic matrix.
     *
     * @tparam T The scalar type of the input and output points (e.g., float, double).
     * @param norm_xy The normalized 2D point represented as an Eigen::Matrix<T, 2, 1>.
     * @return Eigen::Matrix<T, 2, 1> The 2D point in pixel coordinates.
     */
    template<typename T>
    Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,2,1>& norm_xy) const {
        return denormalize(distort(norm_xy));
    }

    /**
     * @brief Projects a 3D point onto a 2D plane using camera intrinsic parameters.
     *
     * This function takes a 3D point in homogeneous coordinates, normalizes it to 2D,
     * applies distortion to the normalized coordinates, and then denormalizes the
     * distorted coordinates to obtain the final 2D projection.
     *
     * @tparam T The scalar type of the input and output (e.g., float, double).
     * @param xyz A 3D point in homogeneous coordinates represented as an Eigen::Matrix<T,3,1>.
     * @return A 2D point in the image plane represented as an Eigen::Matrix<T,2,1>.
     */
    template<typename T>
    Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,3,1>& xyz) const {
        Eigen::Matrix<T,2,1> norm_xy = xyz.hnormalized();
        return denormalize(distort(norm_xy));
    }

    /**
     * @brief Unprojects a 2D pixel coordinate into an undistorted normalized coordinate.
     *
     * This function takes a 2D pixel coordinate, normalizes it, and then applies
     * an undistortion process to return the corresponding undistorted normalized coordinate.
     *
     * @tparam T The scalar type of the input and output coordinates (e.g., float, double).
     * @param pix The 2D pixel coordinate to be unprojected.
     * @return An Eigen::Matrix<T, 2, 1> representing the undistorted normalized coordinate.
     */
    template<typename T>
    Eigen::Matrix<T,2,1> unproject(const Eigen::Matrix<T,2,1>& pix) const {
        return undistort(normalize(pix));
    }
};

} // namespace calib
