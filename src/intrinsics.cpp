#include "calibration/intrinsics.h"

// std
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

// ceres
#include <ceres/ceres.h>

namespace vitavision {

Eigen::Vector2d CameraMatrix::normalize(const Eigen::Vector2d& pix) const {
    return {
        (pix.x() - cx) / fx,
        (pix.y() - cy) / fy
    };
}

Eigen::Vector2d CameraMatrix::denormalize(const Eigen::Vector2d& xy) const {
    return {
        fx * xy.x() + cx,
        fy * xy.y() + cy
    };
}

// Compute a linear least-squares estimate of the camera intrinsics
// (fx, fy, cx, cy) from normalized correspondences.  This ignores lens
// distortion and solves two independent systems:
//   u = fx * x + cx
//   v = fy * y + cy
// If there are insufficient observations or the design matrix is
// degenerate, std::nullopt is returned.
std::optional<CameraMatrix> estimate_intrinsics_linear(
    const std::vector<Observation<double>>& obs,
    std::optional<CalibrationBounds> bounds_opt) {
    if (obs.size() < 2) {
        return std::nullopt;
    }

    // Build separate design matrices for x and y coordinates
    Eigen::MatrixXd Ax(obs.size(), 2);
    Eigen::MatrixXd Ay(obs.size(), 2);
    Eigen::VectorXd bu(obs.size());
    Eigen::VectorXd bv(obs.size());

    for (size_t i = 0; i < obs.size(); ++i) {
        Ax(static_cast<int>(i), 0) = obs[i].x;
        Ax(static_cast<int>(i), 1) = 1.0;

        Ay(static_cast<int>(i), 0) = obs[i].y;
        Ay(static_cast<int>(i), 1) = 1.0;

        bu(static_cast<int>(i)) = obs[i].u;
        bv(static_cast<int>(i)) = obs[i].v;
    }

    // Solve for fx, cx using x coordinates
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_x(Ax, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Detect degenerate configuration
    if (svd_x.singularValues().minCoeff() < 1e-12) {
        return std::nullopt;
    }

    // Solve for fy, cy using y coordinates
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_y(Ay, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Detect degenerate configuration
    if (svd_y.singularValues().minCoeff() < 1e-12) {
        return std::nullopt;
    }

    Eigen::Vector2d xu = svd_x.solve(bu);
    Eigen::Vector2d xv = svd_y.solve(bv);

    CalibrationBounds bounds = bounds_opt.value_or(CalibrationBounds{});

    // Check for reasonably sized intrinsics
    if (xu[0] < bounds.fx_min || xv[0] < bounds.fy_min ||
        xu[0] > bounds.fx_max || xv[0] > bounds.fy_max ||
        xu[1] < bounds.cx_min || xv[1] < bounds.cy_min ||
        xu[1] > bounds.cx_max || xv[1] > bounds.cy_max) {
        std::cerr << "Warning: Linear calibration produced unreasonable intrinsics: fx="
                  << xu[0] << ", fy=" << xv[0] << ", cx=" << xu[1]
                  << ", cy=" << xv[1] << std::endl;
        // Use reasonable defaults based on resolution
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

// Alternate between fitting distortion coefficients and re-estimating
// the camera matrix.  This provides a better linear initialization for
// subsequent non-linear optimization when moderate distortion is
// present.  If the initial linear estimate fails, std::nullopt is
// returned.
std::optional<LinearInitResult> estimate_intrinsics_linear_iterative(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    int max_iterations) {
    // Start with the simple linear estimate that ignores distortion.
    auto K_opt = estimate_intrinsics_linear(obs);
    if (!K_opt) {
        return std::nullopt;
    }
    CameraMatrix K = *K_opt;

    Eigen::VectorXd dist;
    std::vector<Observation<double>> corrected(obs.size());

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Estimate distortion for current intrinsics using original observations.
        dist = fit_distortion(obs, K.fx, K.fy, K.cx, K.cy, num_radial);

        // Remove the estimated distortion from the measurements and
        // re-estimate the intrinsics.
        for (size_t i = 0; i < obs.size(); ++i) {
            Eigen::Vector2d norm(obs[i].x, obs[i].y);
            Eigen::Vector2d distorted = apply_distortion(norm, dist);
            Eigen::Vector2d delta = distorted - norm;
            double u_corr = obs[i].u - K.fx * delta.x();
            double v_corr = obs[i].v - K.fy * delta.y();
            corrected[i] = {obs[i].x, obs[i].y, u_corr, v_corr};
        }

        auto K_new_opt = estimate_intrinsics_linear(corrected);
        if (!K_new_opt) {
            break;
        }
        CameraMatrix K_new = *K_new_opt;

        double diff = std::abs(K_new.fx - K.fx) + std::abs(K_new.fy - K.fy) +
                      std::abs(K_new.cx - K.cx) + std::abs(K_new.cy - K.cy);
        K = K_new;
        if (diff < 1e-6) {
            break;  // Converged
        }
    }

    // Final distortion estimate using refined intrinsics.
    dist = fit_distortion(obs, K.fx, K.fy, K.cx, K.cy, num_radial);

    return LinearInitResult{K, dist};
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

        auto [_, r] = fit_distortion_full(o, intr[0], intr[1], intr[2], intr[3], num_radial_);
        for (int i = 0; i < r.size(); ++i) {
            residuals[i] = r[i];
        }
        return true;
    }

    // Compute optimal distortion coeffs for given intrinsics (useful after
    // Solve).  This is still used by the API after the optimization.
    Eigen::VectorXd SolveDistortionFor(const double intr[4]) const {
        return fit_distortion(obs_, intr[0], intr[1], intr[2], intr[3], num_radial_);
    }
};

static bool compute_covariance(
    size_t n_obs,
    ceres::CostFunction& cost,
    double const* intrinsics,
    ceres::Problem& problem,
    IntrinsicOptimizationResult& result
) {
    // Recompute residuals and RMSE
    const int m = static_cast<int>(n_obs) * 2;
    std::vector<double> residuals(2 * n_obs);

    // Evaluate residuals at the optimum
    const double* params[1] = {intrinsics};
    cost.Evaluate(params, residuals.data(), nullptr);

    double ssr = 0.0;
    for (double r : residuals) ssr += r * r;
    const int dof = m - 4; // 4 nonlinear params in this problem
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

    double cov4x4[16];
    cov.GetCovarianceBlock(intrinsics, intrinsics, cov4x4);

    // Scale by estimated noise variance (unit weights assumption).
    Eigen::Map<Eigen::Matrix4d> covar(cov4x4);
    covar *= sigma2;
    result.covariance = covar;

    return true;
}

IntrinsicOptimizationResult optimize_intrinsics(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verb,
    std::optional<CalibrationBounds> bounds_opt
) {
    double intrinsics[4] = {
        initial_guess.fx,
        initial_guess.fy,
        initial_guess.cx,
        initial_guess.cy
    };

    ceres::Problem problem;
    auto* functor = new IntrinsicsVPResidual(obs, num_radial);
    auto* cost = new ceres::AutoDiffCostFunction<IntrinsicsVPResidual,
                                                 ceres::DYNAMIC, 4>(functor,
                                                                      static_cast<int>(obs.size()) * 2);

    problem.AddResidualBlock(cost, /*loss=*/nullptr, intrinsics);

    CalibrationBounds bounds = bounds_opt.value_or(CalibrationBounds{});
    problem.SetParameterLowerBound(intrinsics, 0, bounds.fx_min);
    problem.SetParameterLowerBound(intrinsics, 1, bounds.fy_min);
    problem.SetParameterLowerBound(intrinsics, 2, bounds.cx_min);
    problem.SetParameterLowerBound(intrinsics, 3, bounds.cy_min);

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
    result.intrinsics.fx = intrinsics[0];
    result.intrinsics.fy = intrinsics[1];
    result.intrinsics.cx = intrinsics[2];
    result.intrinsics.cy = intrinsics[3];
    result.distortion = fit_distortion(
        obs, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], num_radial);
    result.summary = summary.BriefReport();

    // Compute covariance using the optimized cost function
    if (!compute_covariance(obs.size(), *cost, intrinsics, problem, result)) {
        std::cerr << "Covariance computation failed.\n";
    }

    return result;
}

}  // namespace vitavision
