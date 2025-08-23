#pragma once

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calibration/intrinsics.h"
#include "calibration/distortion.h"

namespace vitavision {

// Simple camera model combining intrinsic matrix, distortion coefficients and
// an extrinsic pose representing the transform from the reference/world frame
// to the camera frame.
struct Camera final {
    CameraMatrix intrinsics;       // Camera matrix parameters
    Eigen::VectorXd distortion;    // Distortion coefficients [k..., p1, p2]
    Eigen::Affine3d extrinsic;     // reference -> camera transform

    Camera()
        : intrinsics{0,0,0,0}, distortion(), extrinsic(Eigen::Affine3d::Identity()) {}

    Camera(const CameraMatrix& K, const Eigen::VectorXd& dist,
           const Eigen::Affine3d& ext = Eigen::Affine3d::Identity())
        : intrinsics(K), distortion(dist), extrinsic(ext) {}

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
        Eigen::Matrix<T,2,1> d = apply_distortion(xyn, distortion);
        return intrinsics.denormalize(d);
    }

    /**
     * @brief Get the 3x4 camera projection matrix K[R|t].
     */
    template <typename T = double>
    Eigen::Matrix<T,3,4> projection_matrix() const {
        Eigen::Matrix<T,3,4> Rt;
        Rt.template block<3,3>(0,0) = extrinsic.linear().cast<T>();
        Rt.col(3) = extrinsic.translation().cast<T>();

        Eigen::Matrix<T,3,3> K;
        K << T(intrinsics.fx), T(0), T(intrinsics.cx),
             T(0), T(intrinsics.fy), T(intrinsics.cy),
             T(0), T(0), T(1);
        return K * Rt;
    }

    /**
     * @brief Project a 3D point in the reference/world frame onto the image.
     *
     * The point is first transformed into the camera frame using the extrinsic
     * pose and then projected with the camera intrinsics and distortion model.
     */
    template <typename T>
    Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,3,1>& Pw) const {
        Eigen::Transform<T,3,Eigen::Affine> E = extrinsic.cast<T>();
        Eigen::Matrix<T,3,1> Pc = E * Pw;
        Eigen::Matrix<T,2,1> xyn{Pc.x()/Pc.z(), Pc.y()/Pc.z()};
        return project_normalized(xyn);
    }
};

} // namespace vitavision
