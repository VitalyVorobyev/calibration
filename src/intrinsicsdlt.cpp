#include "calib/intrinsics.h"

// std
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

// ceres
#include <ceres/ceres.h>

namespace calib {

// Compute a linear least-squares estimate of the camera intrinsics
// (fx, fy, cx, cy[, skew]) from normalized correspondences. This ignores lens
// distortion and solves either two or three independent systems depending on
// whether skew is estimated:
//   u = fx * x + skew * y + cx
//   v = fy * y + cy
// If there are insufficient observations or the design matrix is
// degenerate, std::nullopt is returned.
std::optional<CameraMatrix> estimate_intrinsics_linear(const std::vector<Observation<double>>& obs,
                                                       std::optional<CalibrationBounds> bounds_opt,
                                                       bool use_skew) {
    if (obs.size() < 2) {
        return std::nullopt;
    }

    Eigen::MatrixXd ay(obs.size(), 2);
    Eigen::VectorXd bv(obs.size());

    Eigen::VectorXd bu(obs.size());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_x;
    Eigen::VectorXd xu;

    if (use_skew) {
        Eigen::MatrixXd ax(obs.size(), 3);
        for (size_t i = 0; i < obs.size(); ++i) {
            ax(static_cast<int>(i), 0) = obs[i].x;
            ax(static_cast<int>(i), 1) = obs[i].y;
            ax(static_cast<int>(i), 2) = 1.0;
            ay(static_cast<int>(i), 0) = obs[i].y;
            ay(static_cast<int>(i), 1) = 1.0;
            bu(static_cast<int>(i)) = obs[i].u;
            bv(static_cast<int>(i)) = obs[i].v;
        }
        svd_x.compute(ax, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd_x.singularValues().minCoeff() < 1e-12) {
            return std::nullopt;
        }
        xu = svd_x.solve(bu);
    } else {
        Eigen::MatrixXd ax(obs.size(), 2);
        for (size_t i = 0; i < obs.size(); ++i) {
            ax(static_cast<int>(i), 0) = obs[i].x;
            ax(static_cast<int>(i), 1) = 1.0;
            ay(static_cast<int>(i), 0) = obs[i].y;
            ay(static_cast<int>(i), 1) = 1.0;
            bu(static_cast<int>(i)) = obs[i].u;
            bv(static_cast<int>(i)) = obs[i].v;
        }
        svd_x.compute(ax, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd_x.singularValues().minCoeff() < 1e-12) {
            return std::nullopt;
        }
        xu = svd_x.solve(bu);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_y(ay, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd_y.singularValues().minCoeff() < 1e-12) {
        return std::nullopt;
    }

    Eigen::Vector2d xv = svd_y.solve(bv);

    CalibrationBounds bounds = bounds_opt.value_or(CalibrationBounds{});

    if (use_skew) {
        if (xu[0] < bounds.fx_min || xv[0] < bounds.fy_min || xu[0] > bounds.fx_max ||
            xv[0] > bounds.fy_max || xu[2] < bounds.cx_min || xv[1] < bounds.cy_min ||
            xu[2] > bounds.cx_max || xv[1] > bounds.cy_max || xu[1] < bounds.skew_min ||
            xu[1] > bounds.skew_max) {
            std::cerr << "Warning: Linear calibration produced unreasonable intrinsics\n";
            double avg_u = bu.sum() / static_cast<double>(obs.size());
            double avg_v = bv.sum() / static_cast<double>(obs.size());
            xu[0] = std::clamp(std::max(500.0, xu[0]), bounds.fx_min, bounds.fx_max);
            xv[0] = std::clamp(std::max(500.0, xv[0]), bounds.fy_min, bounds.fy_max);
            xu[2] = std::clamp(avg_u / 2.0, bounds.cx_min, bounds.cx_max);
            xv[1] = std::clamp(avg_v / 2.0, bounds.cy_min, bounds.cy_max);
            xu[1] = std::clamp(xu[1], bounds.skew_min, bounds.skew_max);
        }
        CameraMatrix K{xu[0], xv[0], xu[2], xv[1], xu[1]};
        return K;
    } else {
        if (xu[0] < bounds.fx_min || xv[0] < bounds.fy_min || xu[0] > bounds.fx_max ||
            xv[0] > bounds.fy_max || xu[1] < bounds.cx_min || xv[1] < bounds.cy_min ||
            xu[1] > bounds.cx_max || xv[1] > bounds.cy_max) {
            std::cerr << "Warning: Linear calibration produced unreasonable intrinsics\n";
            double avg_u = bu.sum() / static_cast<double>(obs.size());
            double avg_v = bv.sum() / static_cast<double>(obs.size());
            xu[0] = std::clamp(std::max(500.0, xu[0]), bounds.fx_min, bounds.fx_max);
            xv[0] = std::clamp(std::max(500.0, xv[0]), bounds.fy_min, bounds.fy_max);
            xu[1] = std::clamp(avg_u / 2.0, bounds.cx_min, bounds.cx_max);
            xv[1] = std::clamp(avg_v / 2.0, bounds.cy_min, bounds.cy_max);
        }
        CameraMatrix K{xu[0], xv[0], xu[1], xv[1]};
        return K;
    }
}

// Alternate between fitting distortion coefficients and re-estimating
// the camera matrix.  This provides a better linear initialization for
// subsequent non-linear optimization when moderate distortion is
// present.  If the initial linear estimate fails, std::nullopt is
// returned.
std::optional<Camera<BrownConradyd>> estimate_intrinsics_linear_iterative(
    const std::vector<Observation<double>>& obs, int num_radial, int max_iterations,
    bool use_skew) {
    auto kmtx_opt = estimate_intrinsics_linear(obs, std::nullopt, use_skew);
    if (!kmtx_opt.has_value()) {
        return std::nullopt;
    }
    CameraMatrix kmtx = kmtx_opt.value();

    Eigen::VectorXd dist;
    std::vector<Observation<double>> corrected(obs.size());

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Estimate distortion for current intrinsics using original observations.
        auto dist_opt = fit_distortion(obs, kmtx.fx, kmtx.fy, kmtx.cx, kmtx.cy, kmtx.skew, num_radial);
        if (!dist_opt) {
            return std::nullopt;
        }
        dist = dist_opt->distortion;

        // Remove the estimated distortion from the measurements and
        // re-estimate the intrinsics.
        for (size_t i = 0; i < obs.size(); ++i) {
            Eigen::Vector2d norm(obs[i].x, obs[i].y);
            Eigen::Vector2d distorted = apply_distortion(norm, dist);
            Eigen::Vector2d delta = distorted - norm;
            double u_corr = obs[i].u - kmtx.fx * delta.x() - kmtx.skew * delta.y();
            double v_corr = obs[i].v - kmtx.fy * delta.y();
            corrected[i] = {obs[i].x, obs[i].y, u_corr, v_corr};
        }

        auto kmtx_new_opt = estimate_intrinsics_linear(corrected, std::nullopt, use_skew);
        if (!kmtx_new_opt.has_value()) {
            break;
        }
        CameraMatrix kmtx_new = kmtx_new_opt.value();

        double diff = std::abs(kmtx_new.fx - kmtx.fx) + std::abs(kmtx_new.fy - kmtx.fy) +
                      std::abs(kmtx_new.cx - kmtx.cx) + std::abs(kmtx_new.cy - kmtx.cy) +
                      std::abs(kmtx_new.skew - kmtx.skew);
        kmtx = kmtx_new;
        if (diff < 1e-6) {
            break;  // Converged
        }
    }

    auto dual_opt = fit_distortion_full(obs, kmtx.fx, kmtx.fy, kmtx.cx, kmtx.cy, kmtx.skew, num_radial);
    if (!dual_opt) {
        return std::nullopt;
    }
    Camera<BrownConradyd> cam;
    cam.K = kmtx;
    cam.distortion.coeffs = dual_opt->distortion;

    return cam;
}

}  // namespace calib
