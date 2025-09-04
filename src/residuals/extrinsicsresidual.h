/** @brief Residual for joint extrinsic-intrinsic optimization using AnyCamera */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "../observationutils.h"
#include "calib/model/any_camera.h"
#include "calib/planarpose.h"

namespace calib {

template <typename T>
static std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> get_camera_se3_target(
    const Eigen::Matrix<T, 3, 3>& c_R_r, const Eigen::Matrix<T, 3, 1>& c_t_r,
    const Eigen::Matrix<T, 3, 3>& r_R_t, const Eigen::Matrix<T, 3, 1>& r_t_t) {
    auto [c_R_t, c_t_t] = product(c_R_r, c_t_r, r_R_t, r_t_t);
    return {c_R_t, c_t_t};
}

struct ExtrinsicResidual final {
    PlanarView view;
    const AnyCamera& cam;

    ExtrinsicResidual(const PlanarView& v, const AnyCamera& c) : view(v), cam(c) {}

    template <typename T>
    bool operator()(const T* c_q_r, const T* c_t_r, const T* r_q_t, const T* r_t_t,
                    const T* intrinsics, T* residuals) const {
        const auto [c_R_t, c_t_t] =
            get_camera_se3_target(quat_array_to_rotmat(c_q_r), array_to_translation(c_t_r),
                                  quat_array_to_rotmat(r_q_t), array_to_translation(r_t_t));

        size_t idx = 0;
        for (const auto& ob : view) {
            Eigen::Matrix<T, 3, 1> P(T(ob.object_xy.x()), T(ob.object_xy.y()), T(0));
            P = c_R_t * P + c_t_t;
            Eigen::Matrix<T, 2, 1> uv = cam.project<T>(P, intrinsics);
            residuals[idx++] = uv.x() - T(ob.image_uv.x());
            residuals[idx++] = uv.y() - T(ob.image_uv.y());
        }
        return true;
    }

    static ceres::CostFunction* create(const PlanarView& v, const AnyCamera& cam) {
        if (v.empty()) {
            throw std::invalid_argument("No observations provided");
        }
        auto* functor = new ExtrinsicResidual(v, cam);
        int intr_size = cam.traits().param_count;
        return new ceres::AutoDiffCostFunction<ExtrinsicResidual, ceres::DYNAMIC, 4, 3, 4, 3,
                                               ceres::DYNAMIC>(
            functor, static_cast<int>(v.size()) * 2, intr_size);
    }
};

}  // namespace calib
