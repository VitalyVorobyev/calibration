/** @brief Residuals for bundle adjustment with ceres */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "calib/planarpose.h"
#include "calib/camera.h"
#include "calib/scheimpflug.h"
#include "observationutils.h"

namespace calib {

// Computes target -> camera transform
template<typename T>
static std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> get_camera_T_target(
    const Eigen::Matrix<T, 3, 3>& b_R_t, const Eigen::Matrix<T, 3, 1>& b_t_t,
    const Eigen::Matrix<T, 3, 3>& g_R_c, const Eigen::Matrix<T, 3, 1>& g_t_c,
    const Eigen::Matrix<T, 3, 3>& b_R_g, const Eigen::Matrix<T, 3, 1>& b_t_g
) {
    auto [c_R_g, c_t_g] = invert_transform(g_R_c, g_t_c);       // g_T_c -> c_T_g
    auto [g_R_b, g_t_b] = invert_transform(b_R_g, b_t_g);       // b_T_g -> g_T_b
    auto [c_R_b, c_t_b] = product(c_R_g, c_t_g, g_R_b, g_t_b);  // c_T_b = c_T_g * g_T_b
    auto [c_R_t, c_t_t] = product(c_R_b, c_t_b, b_R_t, b_t_t);  // c_T_t = c_T_b * b_T_t
    return {c_R_t, c_t_t};
}

static Eigen::Affine3d get_camera_T_target(
    const Eigen::Affine3d& b_T_t,
    const Eigen::Affine3d& g_T_c,
    const Eigen::Affine3d& b_T_g
) {
    auto c_T_t = g_T_c.inverse() * b_T_g.inverse() * b_T_t;
    return c_T_t;
}

struct BundleReprojResidual final {
    PlanarView view;
    Eigen::Affine3d base_T_gripper;
    BundleReprojResidual(PlanarView v, const Eigen::Affine3d& b_T_g)
        : view(std::move(v)), base_T_gripper(b_T_g) {}

    /**
     * @brief Functor to compute the residuals for a bundle adjustment problem.
     *
     * This operator computes the residuals between observed image points and
     * projected 3D points using the given camera parameters and transformations.
     *
     * @tparam T The scalar type used for computations (e.g., double or ceres::Jet).
     *
     * @param b_q_t Pointer to the quaternion representing the rotation from the base
     *              frame to the target frame.
     * @param b_t_t Pointer to the translation vector from the base frame to the target frame.
     * @param g_q_c Pointer to the quaternion representing the rotation from the gripper
     *              frame to the camera frame.
     * @param g_t_c Pointer to the translation vector from the gripper frame to the camera frame.
     * @param intrinsics Pointer to the array of camera intrinsic parameters and distortion coefficients:
     *                   - intrinsics[0]: Focal length in x direction (fx).
     *                   - intrinsics[1]: Focal length in y direction (fy).
     *                   - intrinsics[2]: Principal point x-coordinate (cx).
     *                   - intrinsics[3]: Principal point y-coordinate (cy).
     *                   - intrinsics[4-8]: Radial and tangential distortion coefficients.
     * @param residuals Pointer to the array where the computed residuals will be stored.
     *                  The residuals are computed as the difference between the observed
     *                  image points and the projected points.
     *
     * @return true Always returns true to indicate successful computation.
     */
    template <typename T>
    bool operator()(const T* b_q_t, const T* b_t_t,
                    const T* g_q_c, const T* g_t_c,
                    const T* intrinsics,
                    T* residuals) const {
        const Eigen::Matrix<T, 3, 3> b_R_g = base_T_gripper.linear().template cast<T>();
        const Eigen::Matrix<T, 3, 1> b_t_g = base_T_gripper.translation().template cast<T>();
        const auto [c_R_t, c_t_t] = get_camera_T_target(
            quat_array_to_rotmat(b_q_t), array_to_translation(b_t_t),
            quat_array_to_rotmat(g_q_c), array_to_translation(g_t_c),
            b_R_g, b_t_g
        );

        CameraMatrixT<T> K{intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]};
        Eigen::Matrix<T,Eigen::Dynamic,1> dist(5);
        dist << intrinsics[4], intrinsics[5], intrinsics[6], intrinsics[7], intrinsics[8];
        Camera<BrownConrady<T>> cam(K, dist);

        size_t idx = 0;
        for (const auto& ob : view) {
            auto P = Eigen::Matrix<T,3,1>(T(ob.object_xy.x()), T(ob.object_xy.y()), T(0));
            P = c_R_t * P + c_t_t;
            Eigen::Matrix<T,2,1> uv = cam.project(P);
            residuals[idx++] = uv.x() - T(ob.image_uv.x());
            residuals[idx++] = uv.y() - T(ob.image_uv.y());
        }
        return true;
    }

    static auto* create(PlanarView v, const Eigen::Affine3d& base_T_gripper) {
        if (v.empty()) {
            throw std::invalid_argument("No observations provided");
        }
        auto functor = new BundleReprojResidual(v, base_T_gripper);
        auto* cost = new ceres::AutoDiffCostFunction<
            BundleReprojResidual, ceres::DYNAMIC,4,3,4,3,9>(
                functor, static_cast<int>(v.size()) * 2);
        return cost;
    }
};

struct BundleScheimpflugReprojResidual final {
    PlanarView view;
    Eigen::Affine3d base_T_gripper;
    BundleScheimpflugReprojResidual(PlanarView v, const Eigen::Affine3d& base_T_gripper)
        : view(std::move(v)), base_T_gripper(base_T_gripper) {}

    /**
     * @brief Functor to compute residuals for bundle adjustment.
     *
     * This operator computes the residuals for a bundle adjustment problem
     * by projecting 3D points onto the image plane using the provided camera
     * and transformation parameters.
     *
     * @tparam T The scalar type, typically `double` or ceres::Jet.
     * @param b_q_t Quaternion representing the rotation from the target to the base.
     * @param b_t_t Translation vector from the target to the base.
     * @param g_q_c Quaternion representing the rotation from the camera to the gripper.
     * @param g_t_c Translation vector from the camera to the gripper.
     * @param intrinsics Camera intrinsics array.
     * @param residuals Output array to store the computed residuals.
     * @return true Always returns true.
     *
     * This function uses the provided transformations to compute the pose of the
     * camera relative to the target. It then projects 3D points from the target
     * frame into the image plane using the camera intrinsics and computes the
     * residuals as the difference between the projected points and the observed
     * image points.
     */
    template <typename T>
    bool operator()(const T* b_q_t, const T* b_t_t,
                    const T* g_q_c, const T* g_t_c,
                    const T* intrinsics,
                    T* residuals) const {
        const Eigen::Matrix<T,3,3> b_R_g = base_T_gripper.linear().template cast<T>();
        const Eigen::Matrix<T,3,1> b_t_g = base_T_gripper.translation().template cast<T>();
        const auto [c_R_t, c_t_t] = get_camera_T_target(
            quat_array_to_rotmat(b_q_t), array_to_translation(b_t_t),
            quat_array_to_rotmat(g_q_c), array_to_translation(g_t_c),
            b_R_g, b_t_g);

        CameraMatrixT<T> K{intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]};
        Eigen::Matrix<T,Eigen::Dynamic,1> dist(5);
        dist << intrinsics[6], intrinsics[7], intrinsics[8], intrinsics[9], intrinsics[10];
        Camera<BrownConrady<T>> cam(K, dist);
        ScheimpflugCamera<BrownConrady<T>> sc(cam, intrinsics[4], intrinsics[5]);

        size_t idx = 0;
        for (const auto& ob : view) {
            auto P = Eigen::Matrix<T,3,1>(T(ob.object_xy.x()), T(ob.object_xy.y()), T(0));
            P = c_R_t * P + c_t_t;
            Eigen::Matrix<T,2,1> uv = sc.project(P);
            residuals[idx++] = uv.x() - T(ob.image_uv.x());
            residuals[idx++] = uv.y() - T(ob.image_uv.y());
        }
        return true;
    }

    static auto* create(PlanarView v, const Eigen::Affine3d& base_T_gripper) {
        auto functor = new BundleScheimpflugReprojResidual(v, base_T_gripper);
        auto* cost = new ceres::AutoDiffCostFunction<
            BundleScheimpflugReprojResidual, ceres::DYNAMIC,4,3,4,3,11>(
                functor, static_cast<int>(v.size()) * 2);
        return cost;
    }
};

}  // namespace calib
