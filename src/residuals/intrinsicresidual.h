/** @brief Residual for camera intrinsics optimization */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "../observationutils.h"
#include "calib/cameramodel.h"
#include "calib/planarpose.h"

namespace calib {

template <camera_model CameraT>
struct IntrinsicResidual final {
    const PlanarView view;

    explicit IntrinsicResidual(PlanarView v) : view(std::move(v)) {}

    template <typename T>
    bool operator()(const T* c_qua_t_, const T* c_tra_t_, const T* intrinsics, T* residuals) const {
        auto c_rot_t = quat_array_to_rotmat(c_qua_t_);
        auto c_tra_t = array_to_translation(c_tra_t_);
        auto cam = CameraTraits<CameraT>::template from_array<T>(intrinsics);

        size_t idx = 0;
        for (const auto& ob : view) {
            auto point = Eigen::Matrix<T, 3, 1>(T(ob.object_xy.x()), T(ob.object_xy.y()), T(0));
            point = c_rot_t * point + c_tra_t;
            Eigen::Matrix<T, 2, 1> uv = cam.project(point);
            residuals[idx++] = uv.x() - T(ob.image_uv.x());
            residuals[idx++] = uv.y() - T(ob.image_uv.y());
        }
        return true;
    }

    static auto* create(const PlanarView& v) {
        if (v.empty()) {
            throw std::invalid_argument("No observations provided");
        };

        auto* functor = new IntrinsicResidual(v);
        constexpr int intr_size = CameraTraits<CameraT>::param_count;
        auto* cost =
            new ceres::AutoDiffCostFunction<IntrinsicResidual, ceres::DYNAMIC, 4, 3, intr_size>(
                functor, static_cast<int>(v.size()) * 2);
        return cost;
    }
};

}  // namespace calib
