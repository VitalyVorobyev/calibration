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
std::optional<LinearInitResult> estimate_intrinsics_linear_iterative(
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

    auto dual_opt = fit_distortion_dual(obs, K.fx, K.fy, K.cx, K.cy, K.skew, num_radial);
    if (!dual_opt) {
        return std::nullopt;
    }
    Camera<DualDistortion> cam;
    cam.K = K;
    cam.distortion = dual_opt->distortion;

    return LinearInitResult{cam};
}

// Residual functor used with AutoDiffCostFunction. The functor performs
// variable projection by solving a linear least squares problem for the
// distortion coefficients for each set of intrinsics.
struct IntrinsicsVPResidual {
    std::vector<Observation<double>> obs_;
    int num_radial_;

    IntrinsicsVPResidual(std::vector<Observation<double>> obs, int num_radial)
        : obs_(std::move(obs)), num_radial_(num_radial) {}

    template <typename T>
    bool operator()(const T* intr, T* residuals) const {
        std::vector<Observation<T>> o(obs_.size());

        std::transform(obs_.begin(), obs_.end(), o.begin(), [](const Observation<double>& obs) {
            return Observation<T>{T(obs.x), T(obs.y), T(obs.u), T(obs.v)};
        });

        auto dr = fit_distortion_full(o, intr[0], intr[1], intr[2], intr[3], intr[4], num_radial_);
        if (!dr) return false;
        const auto& r = dr->residuals;
        for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        return true;
    }

    // Compute optimal distortion coeffs for given intrinsics (useful after
    // Solve).  This is still used by the API after the optimization.
    Eigen::VectorXd SolveDistortionFor(const double intr[5]) const {
        auto d = fit_distortion(obs_, intr[0], intr[1], intr[2], intr[3], intr[4], num_radial_);
        return d ? d->distortion : Eigen::VectorXd{};
    }

    static auto create_cost(const std::vector<Observation<double>>& obs,
                            int num_radial) {
        auto* functor = new IntrinsicsVPResidual(obs, num_radial);
        auto* cost = new ceres::AutoDiffCostFunction<IntrinsicsVPResidual,
            ceres::DYNAMIC, 5>(functor, static_cast<int>(obs.size()) * 2);
        return cost;
    }
};

static bool compute_covariance(
    size_t n_obs,
    ceres::CostFunction& cost,
    double const* intrinsics,
    ceres::Problem& problem,
    IntrinsicOptimizationResult& result,
    int param_dim
) {
    // Recompute residuals and RMSE
    const int m = static_cast<int>(n_obs) * 2;
    std::vector<double> residuals(2 * n_obs);

    // Evaluate residuals at the optimum
    const double* params[1] = {intrinsics};
    cost.Evaluate(params, residuals.data(), nullptr);

    double ssr = 0.0;
    for (double r : residuals) ssr += r * r;
    const int dof = m - param_dim;
    result.reprojection_error = std::sqrt(ssr / m);
    const double sigma2 = ssr / std::max(1, dof);

    // Covariance (approximate): scale by residual variance.
    ceres::Covariance::Options cov_opts;
    ceres::Covariance cov(cov_opts);
    std::vector<std::pair<const double*, const double*>> blocks = { {intrinsics, intrinsics} };
    if (!cov.Compute(blocks, &problem)) {
        std::cerr << "Covariance computation failed.\n";
        return false;
    }

    double cov5x5[25];
    cov.GetCovarianceBlock(intrinsics, intrinsics, cov5x5);

    Eigen::Map<Eigen::Matrix<double,5,5>> covar(cov5x5);
    covar *= sigma2;
    result.covariance = covar;

    return true;
}

IntrinsicOptimizationResult optimize_intrinsics(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verb,
    std::optional<CalibrationBounds> bounds_opt,
    bool use_skew
) {
    double intrinsics[5] = {
        initial_guess.fx,
        initial_guess.fy,
        initial_guess.cx,
        initial_guess.cy,
        initial_guess.skew
    };

    ceres::Problem problem;
    auto* cost = IntrinsicsVPResidual::create_cost(obs, num_radial);
    problem.AddResidualBlock(cost, /*loss=*/nullptr, intrinsics);

    CalibrationBounds bounds = bounds_opt.value_or(CalibrationBounds{});
    problem.SetParameterLowerBound(intrinsics, 0, bounds.fx_min);
    problem.SetParameterLowerBound(intrinsics, 1, bounds.fy_min);
    problem.SetParameterLowerBound(intrinsics, 2, bounds.cx_min);
    problem.SetParameterLowerBound(intrinsics, 3, bounds.cy_min);
    if (use_skew) {
        problem.SetParameterLowerBound(intrinsics, 4, bounds.skew_min);
        problem.SetParameterUpperBound(intrinsics, 4, bounds.skew_max);
    } else {
        problem.SetManifold(intrinsics, new ceres::SubsetManifold(5, {4}));
    }

    problem.SetParameterUpperBound(intrinsics, 0, bounds.fx_max);
    problem.SetParameterUpperBound(intrinsics, 1, bounds.fy_max);
    problem.SetParameterUpperBound(intrinsics, 2, bounds.cx_max);
    problem.SetParameterUpperBound(intrinsics, 3, bounds.cy_max);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = verb;
    options.function_tolerance = 1e-12;
    options.gradient_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    IntrinsicOptimizationResult result;
    result.camera.K.fx = intrinsics[0];
    result.camera.K.fy = intrinsics[1];
    result.camera.K.cx = intrinsics[2];
    result.camera.K.cy = intrinsics[3];
    result.camera.K.skew = intrinsics[4];

    auto dual_opt = fit_distortion_dual(
        obs, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], intrinsics[4], num_radial);
    if (dual_opt) {
        result.camera.distortion = dual_opt->distortion;
    }
    result.summary = summary.BriefReport();

    int param_dim = use_skew ? 5 : 4;
    if (!compute_covariance(obs.size(), *cost, intrinsics, problem, result, param_dim)) {
        std::cerr << "Covariance computation failed.\n";
    }

    return result;
}

}  // namespace calib
