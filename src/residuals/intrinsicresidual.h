/** @brief Residual for camera intrinsics optimization using AnyCamera */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "../observationutils.h"
#include "calib/model/any_camera.h"
#include "calib/planarpose.h"

namespace calib {

struct IntrinsicResidual final {
  PlanarView view;
  const AnyCamera& cam;

  IntrinsicResidual(const PlanarView& v, const AnyCamera& c) : view(v), cam(c) {}

  template <typename T>
  bool operator()(const T* c_q_t_, const T* c_t_t_, const T* intrinsics,
                  T* residuals) const {
    auto c_R_t = quat_array_to_rotmat(c_q_t_);
    auto c_t_t = array_to_translation(c_t_t_);

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
    auto* functor = new IntrinsicResidual(v, cam);
    int intr_size = cam.traits().param_count;
    return new ceres::AutoDiffCostFunction<IntrinsicResidual, ceres::DYNAMIC, 4, 3,
                                           ceres::DYNAMIC>(functor,
                                                            static_cast<int>(v.size()) * 2,
                                                            intr_size);
  }
};

}  // namespace calib
