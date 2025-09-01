/** @brief Semi-linear residual for intrinsic camera parameters */

#pragma once

// std
#include <vector>

// ceres
#include <ceres/ceres.h>

#include "calib/planarpose.h"
#include "calib/distortion.h"

namespace calib {

// Variable projection residual for full camera calibration.
struct CalibVPResidual final {
    const std::vector<PlanarView> views;  // observations per view
    int num_radial_;
    size_t total_obs_;

    CalibVPResidual(const std::vector<PlanarView>& v, int num_radial)
        : views(v), num_radial_(num_radial) {
        total_obs_ = 0;
        for (const auto& view : v) total_obs_ += view.size();
    }

    template<typename T>
    bool operator()(T const* const* params, T* residuals) const {

        std::vector<Observation<T>> o;
        o.reserve(total_obs_);

        auto c_T_t = Eigen::Transform<T, 3, Eigen::Affine>::Identity();

        for (size_t i = 0; i < views_.size(); ++i) {
            c_T_t.linear() = quat_array_to_rotmat<T>(params[2 * i + 1]);
            c_T_t.translation() = array_to_translation<T>(params[2 * i + 2]);
            std::vector<Observation<T>> new_obs(views_[i].size());
            planar_observables_to_observables(views_[i], new_obs, c_T_t);
            o.insert(o.end(), new_obs.begin(), new_obs.end());
        }

        const T* intr = params[0];
        auto dr = fit_distortion_full(o, intr[0], intr[1], intr[2], intr[3], intr[4], num_radial_);
        if (!dr) return false;
        const auto& r = dr->residuals;
        for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        return true;
    }

    static auto* create(const std::vector<PlanarView>& views, int num_radial) {
        auto* functor = new CalibVPResidual(views, num_radial);
        auto* cost = new ceres::DynamicAutoDiffCostFunction(functor);
        cost->AddParameterBlock(5);  // Intrinsics (fx, fy, cx, cy, skew)
        for (size_t i = 0; i < views.size(); ++i) {
            cost->AddParameterBlock(4);  // Quaternion for each view
            cost->AddParameterBlock(3);  // Translation for each view
        }
        size_t total_obs = 0;
        for (const auto& v : views) total_obs += v.size();
        cost->SetNumResiduals(static_cast<int>(total_obs * 2));
        return cost;
    }
};

}  // namespace calib
