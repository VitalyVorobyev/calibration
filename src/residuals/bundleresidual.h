/** @brief Residuals for bundle adjustment with ceres */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "../observationutils.h"
#include "calib/cameramodel.h"
#include "calib/planarpose.h"

namespace calib {

// Computes target -> camera transform
template <typename T>
static std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> get_camera_se3_target(
    const Eigen::Matrix<T, 3, 3>& b_rot_t, const Eigen::Matrix<T, 3, 1>& b_tra_t,
    const Eigen::Matrix<T, 3, 3>& g_rot_c, const Eigen::Matrix<T, 3, 1>& g_tra_c,
    const Eigen::Matrix<T, 3, 3>& b_rot_g, const Eigen::Matrix<T, 3, 1>& b_tra_g) {
    auto [c_rot_g, c_tra_g] = invert_transform(g_rot_c, g_tra_c);  // g_se3_c -> c_se3_g
    auto [g_rot_b, g_tra_b] = invert_transform(b_rot_g, b_tra_g);  // b_se3_g -> g_se3_b
    auto [c_rot_b, c_tra_b] =
        product(c_rot_g, c_tra_g, g_rot_b, g_tra_b);  // c_se3_b = c_se3_g * g_se3_b
    auto [c_rot_t, c_tra_t] =
        product(c_rot_b, c_tra_b, b_rot_t, b_tra_t);  // c_se3_t = c_se3_b * b_se3_t
    return {c_rot_t, c_tra_t};
}

#if 0
static Eigen::Isometry3d get_camera_se3_target(
    const Eigen::Isometry3d& b_se3_t,
    const Eigen::Isometry3d& g_se3_c,
    const Eigen::Isometry3d& b_se3_g
) {
    auto c_se3_t = g_se3_c.inverse() * b_se3_g.inverse() * b_se3_t;
    return c_se3_t;
}
#endif

template <camera_model CameraT>
struct BundleReprojResidual final {
    const PlanarView view;
    const Eigen::Isometry3d base_se3_gripper;
    BundleReprojResidual(const PlanarView& v, const Eigen::Isometry3d& b_se3_g)
        : view(v), base_se3_gripper(b_se3_g) {}

    template <typename T>
    bool operator()(const T* b_quat_t, const T* b_tra_t, const T* g_quat_c, const T* g_tra_c,
                    const T* intrinsics, T* residuals) const {
        const Eigen::Matrix<T, 3, 3> b_rot_g = base_se3_gripper.linear().template cast<T>();
        const Eigen::Matrix<T, 3, 1> b_tra_g = base_se3_gripper.translation().template cast<T>();
        const auto [c_rot_t, c_tra_t] = get_camera_se3_target(
            quat_array_to_rotmat(b_quat_t), array_to_translation(b_tra_t),
            quat_array_to_rotmat(g_quat_c), array_to_translation(g_tra_c), b_rot_g, b_tra_g);

        auto cam = CameraTraits<CameraT>::template from_array<T>(intrinsics);

        size_t idx = 0;
        for (const auto& ob : view) {
            auto P = Eigen::Matrix<T, 3, 1>(T(ob.object_xy.x()), T(ob.object_xy.y()), T(0));
            P = c_rot_t * P + c_tra_t;
            Eigen::Matrix<T, 2, 1> uv = cam.project(P);
            residuals[idx++] = uv.x() - T(ob.image_uv.x());
            residuals[idx++] = uv.y() - T(ob.image_uv.y());
        }
        return true;
    }

    static auto* create(const PlanarView& v, const Eigen::Isometry3d& base_se3_gripper) {
        if (v.empty()) {
            throw std::invalid_argument("No observations provided");
        }
        auto* functor = new BundleReprojResidual(v, base_se3_gripper);
        constexpr int intr_size = CameraTraits<CameraT>::param_count;
        auto* cost = new ceres::AutoDiffCostFunction<BundleReprojResidual, ceres::DYNAMIC, 4, 3, 4,
                                                     3, intr_size>(
            functor, static_cast<int>(functor->view.size()) * 2);
        return cost;
    }
};

}  // namespace calib
