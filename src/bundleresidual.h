/** @brief Residuals for bundle adjustment with ceres */

#pragma once

// ceres
#include <ceres/ceres.h>

#include "calibration/planarpose.h"

namespace vitavision {

struct HandEyeReprojResidual final {
    PlanarView view;
    Eigen::Affine3d base_to_gripper;
    HandEyeReprojResidual(PlanarView v, const Eigen::Affine3d& base_T_gripper)
        : view(std::move(v)), base_to_gripper(base_T_gripper) {}

    template <typename T>
    bool operator()(const T* base_target6, const T* he_ref6, const T* ext6,
                    const T* intrinsics, const T* dist, T* residuals) const {
        auto base_T_target = pose2affine(base_target6);  // target -> base
        auto refcam_T_gripper = pose2affine(he_ref6);    // gripper -> reference camera
        auto camera_T_refcam = pose2affine(ext6);        // reference -> camera extrinsic
        auto camera_T_target = get_camera_T_target(
            base_T_target, refcam_T_gripper, camera_T_refcam, base_to_gripper.template cast<T>());

        std::vector<Observation<T>> o(view.size());
        planar_observables_to_observables(view, o, camera_T_target);

        const T fx = intrinsics[0];
        const T fy = intrinsics[1];
        const T cx = intrinsics[2];
        const T cy = intrinsics[3];
        Eigen::Map<const Eigen::Matrix<T,Eigen::Dynamic,1>> d(dist, 4);

        int idx = 0;
        for (const auto& ob : o) {
            Eigen::Matrix<T,2,1> norm_xy(ob.x, ob.y);
            Eigen::Matrix<T,2,1> distorted = apply_distortion<T>(norm_xy, d);
            T u = fx * distorted.x() + cx;
            T v = fy * distorted.y() + cy;
            residuals[idx++] = u - ob.u;
            residuals[idx++] = v - ob.v;
        }
        return true;
    }

    static auto* create(PlanarView v, const Eigen::Affine3d& base_T_gripper) {
        auto functor = new HandEyeReprojResidual(v, base_T_gripper);
        auto* cost = new ceres::AutoDiffCostFunction<
            HandEyeReprojResidual, ceres::DYNAMIC, 6,6,6,4,4>(
                functor, static_cast<int>(v.size()) * 2);
        return cost;
    }
};

}  // namespace vitavision
