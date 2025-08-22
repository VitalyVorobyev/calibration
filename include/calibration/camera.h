#pragma once

// eigen
#include <Eigen/Core>

#include "calibration/intrinsics.h"
#include "calibration/distortion.h"

namespace vitavision {

// Simple camera model combining intrinsic matrix and distortion coefficients.
struct Camera {
    CameraMatrix intrinsics;      // Camera matrix parameters
    Eigen::VectorXd distortion;   // Distortion coefficients [k..., p1, p2]

    Camera() = default;
    Camera(const CameraMatrix& K, const Eigen::VectorXd& dist)
        : intrinsics(K), distortion(dist) {}

    template <typename T>
    Eigen::Matrix<T,2,1> projectNormalized(const Eigen::Matrix<T,2,1>& xyn) const {
        Eigen::Matrix<T,2,1> d = apply_distortion(xyn, distortion);
        return intrinsics.denormalize(d);
    }
};

} // namespace vitavision

