#pragma once

// eigen
#include <Eigen/Core>

#include "calibration/intrinsics.h"
#include "calibration/distortion.h"

namespace vitavision {

// Simple camera model combining intrinsic matrix and distortion coefficients.
struct Camera final {
    CameraMatrix intrinsics;      // Camera matrix parameters
    Eigen::VectorXd distortion;   // Distortion coefficients [k..., p1, p2]

    Camera() = default;
    Camera(const CameraMatrix& K, const Eigen::VectorXd& dist)
        : intrinsics(K), distortion(dist) {}

    /**
     * @brief Projects a 2D point in normalized coordinates to pixel coordinates.
     *
     * This function applies the camera's distortion model to the input normalized
     * coordinates and then converts the distorted coordinates to pixel coordinates
     * using the camera's intrinsic parameters.
     *
     * @tparam T The scalar type of the input and output coordinates (e.g., float, double).
     * @param xyn A 2D point in normalized image coordinates.
     * @return A 2D point in pixel coordinates after applying distortion and denormalization.
     */
    template <typename T>
    Eigen::Matrix<T,2,1> project_normalized(const Eigen::Matrix<T,2,1>& xyn) const {
        Eigen::Matrix<T, Eigen::Dynamic, 1> distT = distortion.template cast<T>();
        Eigen::Matrix<T,2,1> d = apply_distortion(xyn, distT);
        return intrinsics.denormalize(d);
    }
};

} // namespace vitavision
