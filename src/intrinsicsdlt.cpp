#include "calib/intrinsics.h"

// std
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

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
std::optional<CameraMatrix> estimate_intrinsics_linear(
    const std::vector<Observation<double>>& obs,
    std::optional<CalibrationBounds> bounds_opt,
    bool use_skew) {
    if (obs.size() < 2) {
        return std::nullopt;
    }

    Eigen::MatrixXd Ay(obs.size(), 2);
    Eigen::VectorXd bv(obs.size());

    Eigen::VectorXd bu(obs.size());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_x;
    Eigen::VectorXd xu;

    if (use_skew) {
        Eigen::MatrixXd Ax(obs.size(), 3);
        for (size_t i = 0; i < obs.size(); ++i) {
            Ax(static_cast<int>(i), 0) = obs[i].x;
            Ax(static_cast<int>(i), 1) = obs[i].y;
            Ax(static_cast<int>(i), 2) = 1.0;
            Ay(static_cast<int>(i), 0) = obs[i].y;
            Ay(static_cast<int>(i), 1) = 1.0;
            bu(static_cast<int>(i)) = obs[i].u;
            bv(static_cast<int>(i)) = obs[i].v;
        }
        svd_x.compute(Ax, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd_x.singularValues().minCoeff() < 1e-12) {
            return std::nullopt;
        }
        xu = svd_x.solve(bu);
    } else {
        Eigen::MatrixXd Ax(obs.size(), 2);
        for (size_t i = 0; i < obs.size(); ++i) {
            Ax(static_cast<int>(i), 0) = obs[i].x;
            Ax(static_cast<int>(i), 1) = 1.0;
            Ay(static_cast<int>(i), 0) = obs[i].y;
            Ay(static_cast<int>(i), 1) = 1.0;
            bu(static_cast<int>(i)) = obs[i].u;
            bv(static_cast<int>(i)) = obs[i].v;
        }
        svd_x.compute(Ax, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd_x.singularValues().minCoeff() < 1e-12) {
            return std::nullopt;
        }
        xu = svd_x.solve(bu);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_y(Ay, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd_y.singularValues().minCoeff() < 1e-12) {
        return std::nullopt;
    }

    Eigen::Vector2d xv = svd_y.solve(bv);

    CalibrationBounds bounds = bounds_opt.value_or(CalibrationBounds{});

    if (use_skew) {
        if (xu[0] < bounds.fx_min || xv[0] < bounds.fy_min ||
            xu[0] > bounds.fx_max || xv[0] > bounds.fy_max ||
            xu[2] < bounds.cx_min || xv[1] < bounds.cy_min ||
            xu[2] > bounds.cx_max || xv[1] > bounds.cy_max ||
            xu[1] < bounds.skew_min || xu[1] > bounds.skew_max) {
            std::cerr << "Warning: Linear calibration produced unreasonable intrinsics\n";
            double avg_u = bu.sum() / obs.size();
            double avg_v = bv.sum() / obs.size();
            xu[0] = std::clamp(std::max(500.0, xu[0]), bounds.fx_min, bounds.fx_max);
            xv[0] = std::clamp(std::max(500.0, xv[0]), bounds.fy_min, bounds.fy_max);
            xu[2] = std::clamp(avg_u / 2.0, bounds.cx_min, bounds.cx_max);
            xv[1] = std::clamp(avg_v / 2.0, bounds.cy_min, bounds.cy_max);
            xu[1] = std::clamp(xu[1], bounds.skew_min, bounds.skew_max);
        }
        CameraMatrix K{xu[0], xv[0], xu[2], xv[1], xu[1]};
        return K;
    } else {
        if (xu[0] < bounds.fx_min || xv[0] < bounds.fy_min ||
            xu[0] > bounds.fx_max || xv[0] > bounds.fy_max ||
            xu[1] < bounds.cx_min || xv[1] < bounds.cy_min ||
            xu[1] > bounds.cx_max || xv[1] > bounds.cy_max) {
            std::cerr << "Warning: Linear calibration produced unreasonable intrinsics\n";
            double avg_u = bu.sum() / obs.size();
            double avg_v = bv.sum() / obs.size();
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
    const std::vector<Observation<double>>& obs,
    int num_radial,
    int max_iterations,
    bool use_skew) {
    auto K_opt = estimate_intrinsics_linear(obs, std::nullopt, use_skew);
    if (!K_opt) {
        return std::nullopt;
    }
    CameraMatrix K = *K_opt;

    Eigen::VectorXd dist;
    std::vector<Observation<double>> corrected(obs.size());

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Estimate distortion for current intrinsics using original observations.
        auto dist_opt = fit_distortion(obs, K.fx, K.fy, K.cx, K.cy, K.skew, num_radial);
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
            double u_corr = obs[i].u - K.fx * delta.x() - K.skew * delta.y();
            double v_corr = obs[i].v - K.fy * delta.y();
            corrected[i] = {obs[i].x, obs[i].y, u_corr, v_corr};
        }

        auto K_new_opt = estimate_intrinsics_linear(corrected, std::nullopt, use_skew);
        if (!K_new_opt) {
            break;
        }
        CameraMatrix K_new = *K_new_opt;

        double diff = std::abs(K_new.fx - K.fx) + std::abs(K_new.fy - K.fy) +
                      std::abs(K_new.cx - K.cx) + std::abs(K_new.cy - K.cy) +
                      std::abs(K_new.skew - K.skew);
        K = K_new;
        if (diff < 1e-6) {
            break;  // Converged
        }
    }

    auto dual_opt = fit_distortion_full(obs, K.fx, K.fy, K.cx, K.cy, K.skew, num_radial);
    if (!dual_opt) {
        return std::nullopt;
    }
    Camera<BrownConradyd> cam;
    cam.K = K;
    cam.distortion.coeffs = dual_opt->distortion;

    return cam;
}

}  // namespace calib
