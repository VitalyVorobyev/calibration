/** @brief Residual for joint extrinsic-intrinsic optimization */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "../detail/observationutils.h"
#include "calib/estimation/planarpose.h"
#include "calib/models/cameramodel.h"

namespace calib {

template <typename T>
static std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> get_camera_se3_target(
    const Eigen::Matrix<T, 3, 3>& c_rot_r, const Eigen::Matrix<T, 3, 1>& c_tra_r,
    const Eigen::Matrix<T, 3, 3>& r_rot_t, const Eigen::Matrix<T, 3, 1>& r_tra_t) {
    auto [c_rot_t, c_tra_t] = product(c_rot_r, c_tra_r, r_rot_t, r_tra_t);
    return {c_rot_t, c_tra_t};
}

template <camera_model CameraT>
struct ExtrinsicResidual final {
    const PlanarView view;

    explicit ExtrinsicResidual(PlanarView view_) : view(std::move(view_)) {}

    template <typename T>
    bool operator()(const T* c_qua_r, const T* c_tra_r, const T* r_qua_t, const T* r_tra_t,
                    const T* intrinsics, T* residuals) const {
        const auto [c_rot_t, c_tra_t] =
            get_camera_se3_target(quat_array_to_rotmat(c_qua_r), array_to_translation(c_tra_r),
                                  quat_array_to_rotmat(r_qua_t), array_to_translation(r_tra_t));
        auto cam = CameraTraits<CameraT>::template from_array<T>(intrinsics);

        size_t idx = 0;
        for (const auto& observation : view) {
            auto point = Eigen::Matrix<T, 3, 1>(T(observation.object_xy.x()),
                                                T(observation.object_xy.y()), T(0));
            point = c_rot_t * point + c_tra_t;
            Eigen::Matrix<T, 2, 1> projected_uv = cam.project(point);
            residuals[idx++] = projected_uv.x() - T(observation.image_uv.x());
            residuals[idx++] = projected_uv.y() - T(observation.image_uv.y());
        }
        return true;
    }

    static auto* create(const PlanarView& view) {
        if (view.empty()) {
            throw std::invalid_argument("No observations provided");
        }

        auto* functor = new ExtrinsicResidual(view);
        constexpr int intr_size = CameraTraits<CameraT>::param_count;
        auto* cost =
            new ceres::AutoDiffCostFunction<ExtrinsicResidual, ceres::DYNAMIC, 4, 3, 4, 3,
                                            intr_size>(functor, static_cast<int>(view.size()) * 2);
        return cost;
    }
};

}  // namespace calib
