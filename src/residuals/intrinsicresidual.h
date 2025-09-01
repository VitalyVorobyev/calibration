/** @brief Residual for camera intrinsics optimization */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "calib/planarpose.h"
#include "calib/cameramodel.h"

#include "../observationutils.h"

namespace calib {

template<camera_model CameraT>
struct IntrinsicResidual final {
    const PlanarView view;

    explicit IntrinsicResidual(const PlanarView& v) : view(v) {}

    template <typename T>
    bool operator()(const T* c_q_t_, const T* c_t_t_,
                    const T* intrinsics, T* residuals) const {
        auto c_R_t = quat_array_to_rotmat(c_q_t_);
        auto c_t_t = array_to_translation(c_t_t_);
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

    static auto* create(const PlanarView& v) {
        if (v.empty()) throw std::invalid_argument("No observations provided");

        auto* functor = new IntrinsicResidual(v);
        constexpr int intr_size = CameraTraits<CameraT>::param_count;
        auto* cost = new ceres::AutoDiffCostFunction<
            IntrinsicResidual, ceres::DYNAMIC,4,3,intr_size>(
                functor, static_cast<int>(v.size()) * 2);
        return cost;
    }
};

}  // namespace calib
