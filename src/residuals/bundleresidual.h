/** @brief Residuals for bundle adjustment using AnyCamera */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "../observationutils.h"
#include "calib/model/any_camera.h"
#include "calib/planarpose.h"

namespace calib {

// Computes target -> camera transform
template <typename T>
static std::pair<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 1>> get_camera_se3_target(
    const Eigen::Matrix<T, 3, 3>& b_R_t, const Eigen::Matrix<T, 3, 1>& b_t_t,
    const Eigen::Matrix<T, 3, 3>& g_R_c, const Eigen::Matrix<T, 3, 1>& g_t_c,
    const Eigen::Matrix<T, 3, 3>& b_R_g, const Eigen::Matrix<T, 3, 1>& b_t_g) {
  auto [c_R_g, c_t_g] = invert_transform(g_R_c, g_t_c);       // g_se3_c -> c_se3_g
  auto [g_R_b, g_t_b] = invert_transform(b_R_g, b_t_g);       // b_se3_g -> g_se3_b
  auto [c_R_b, c_t_b] = product(c_R_g, c_t_g, g_R_b, g_t_b);  // c_se3_b = c_se3_g * g_se3_b
  auto [c_R_t, c_t_t] = product(c_R_b, c_t_b, b_R_t, b_t_t);  // c_se3_t = c_se3_b * b_se3_t
  return {c_R_t, c_t_t};
}

struct BundleReprojResidual final {
  PlanarView view;
  Eigen::Isometry3d base_se3_gripper;
  const AnyCamera& cam;

  BundleReprojResidual(const PlanarView& v, const Eigen::Isometry3d& b_se3_g,
                       const AnyCamera& c)
      : view(v), base_se3_gripper(b_se3_g), cam(c) {}

  template <typename T>
  bool operator()(const T* b_q_t, const T* b_t_t, const T* g_q_c, const T* g_t_c,
                  const T* intrinsics, T* residuals) const {
    const Eigen::Matrix<T, 3, 3> b_R_g = base_se3_gripper.linear().template cast<T>();
    const Eigen::Matrix<T, 3, 1> b_t_g = base_se3_gripper.translation().template cast<T>();
    const auto [c_R_t, c_t_t] = get_camera_se3_target(
        quat_array_to_rotmat(b_q_t), array_to_translation(b_t_t),
        quat_array_to_rotmat(g_q_c), array_to_translation(g_t_c), b_R_g, b_t_g);

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

  static ceres::CostFunction* create(const PlanarView& v,
                                     const Eigen::Isometry3d& b_se3_g,
                                     const AnyCamera& cam) {
    if (v.empty()) {
      throw std::invalid_argument("No observations provided");
    }
    auto* functor = new BundleReprojResidual(v, b_se3_g, cam);
    int intr_size = cam.traits().param_count;
    return new ceres::AutoDiffCostFunction<BundleReprojResidual, ceres::DYNAMIC, 4, 3, 4,
                                           3, ceres::DYNAMIC>(functor,
                                                              static_cast<int>(v.size()) * 2,
                                                              intr_size);
  }
};

}  // namespace calib
