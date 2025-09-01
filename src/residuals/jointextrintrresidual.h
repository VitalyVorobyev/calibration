/** @brief Residual for joint extrinsic-intrinsic optimization */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "calib/planarpose.h"
#include "calib/cameramodel.h"

#include "../observationutils.h"

namespace calib {

template<typename T>
static std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> get_camera_T_target(
    const Eigen::Matrix<T, 3, 3>& c_R_r, const Eigen::Matrix<T, 3, 1>& c_t_r,
    const Eigen::Matrix<T, 3, 3>& r_R_t, const Eigen::Matrix<T, 3, 1>& r_t_t
) {
    auto [c_R_t, c_t_t] = product(c_R_r, c_t_r, r_R_t, r_t_t);
    return {c_R_t, c_t_t};
}

template<camera_model CameraT>
struct JointIntrExtrResidual final {
    const PlanarView view;

    JointIntrExtrResidual(PlanarView&& v) : view(std::move(v)) {}

    template <typename T>
    bool operator()(const T* c_q_r, const T* c_t_r,
                    const T* r_q_t, const T* r_t_t,
                    const T* intrinsics, T* residuals) const {
        const auto [c_R_t, c_t_t] = get_camera_T_target(
            quat_array_to_rotmat(c_q_r), array_to_translation(c_t_r),
            quat_array_to_rotmat(r_q_t), array_to_translation(r_t_t)
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
    }

    static auto* create(PlanarView&& view) {
        if (view.empty()) throw std::invalid_argument("No observations provided");

        auto* functor = new JointIntrExtrResidual(std::move(view));
        constexpr int intr_size = CameraTraits<CameraT>::param_count;
        auto* cost = new ceres::AutoDiffCostFunction<
            JointIntrExtrResidual, ceres::DYNAMIC,4,3,4,3,intr_size>(
                functor, static_cast<int>(obs.size()) * 2);
        return cost;
    }
};

}  // namespace calib
