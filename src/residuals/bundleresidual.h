/** @brief Residuals for bundle adjustment with ceres */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "calib/planarpose.h"
#include "calib/cameramodel.h"

#include "../observationutils.h"

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

#if 0
static Eigen::Affine3d get_camera_T_target(
    const Eigen::Affine3d& b_T_t,
    const Eigen::Affine3d& g_T_c,
    const Eigen::Affine3d& b_T_g
) {
    auto c_T_t = g_T_c.inverse() * b_T_g.inverse() * b_T_t;
    return c_T_t;
}
#endif

template<camera_model CameraT>
struct BundleReprojResidual final {
    const PlanarView view;
    const Eigen::Affine3d base_T_gripper;
    BundleReprojResidual(const PlanarView& v, const Eigen::Affine3d& b_T_g)
        : view(v), base_T_gripper(b_T_g) {}

    template <typename T>
    bool operator()(const T* b_q_t, const T* b_t_t,
                    const T* g_q_c, const T* g_t_c,
                    const T* intrinsics, T* residuals) const {
        const Eigen::Matrix<T, 3, 3> b_R_g = base_T_gripper.linear().template cast<T>();
        const Eigen::Matrix<T, 3, 1> b_t_g = base_T_gripper.translation().template cast<T>();
        const auto [c_R_t, c_t_t] = get_camera_T_target(
            quat_array_to_rotmat(b_q_t), array_to_translation(b_t_t),
            quat_array_to_rotmat(g_q_c), array_to_translation(g_t_c),
            b_R_g, b_t_g
        );

        auto cam = CameraTraits<CameraT>::template from_array<T>(intrinsics);

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

    static auto* create(const PlanarView& v, const Eigen::Affine3d& base_T_gripper) {
        if (v.empty()) {
            throw std::invalid_argument("No observations provided");
        }
        auto* functor = new BundleReprojResidual(v, base_T_gripper);
        constexpr int intr_size = CameraTraits<CameraT>::param_count;
        auto* cost = new ceres::AutoDiffCostFunction<
            BundleReprojResidual, ceres::DYNAMIC,4,3,4,3,intr_size>(
                functor, static_cast<int>(functor->view.size()) * 2);
        return cost;
    }
};

}  // namespace calib
