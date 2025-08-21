#include "calibration/intrinsics.h"

// std
#include <iostream>
#include <random>
#include <cmath>

// ceres
#include <ceres/ceres.h>

namespace vitavision {

// Residual functor used with AutoDiffCostFunction. The functor performs
// variable projection by solving a linear least squares problem for the
// distortion coefficients for each set of intrinsics.
struct IntrinsicsVPResidual {
    std::vector<Observation> obs_;
    int num_radial_;

    IntrinsicsVPResidual(std::vector<Observation> obs, int num_radial)
        : obs_(std::move(obs)), num_radial_(num_radial) {}

    template <typename T>
    bool operator()(const T* intr, T* residuals) const {
        const int M = num_radial_ + 2;               // radial + tangential coeffs
        const int rows = static_cast<int>(obs_.size()) * 2;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A(rows, M);
        Eigen::Matrix<T, Eigen::Dynamic, 1> b(rows);

        for (int i = 0, n = static_cast<int>(obs_.size()); i < n; ++i) {
            const T x = T(obs_[i].x);
            const T y = T(obs_[i].y);
            const T r2 = x * x + y * y;

            const T u0 = intr[0] * x + intr[2];
            const T v0 = intr[1] * y + intr[3];

            const T du = T(obs_[i].u) - u0;
            const T dv = T(obs_[i].v) - v0;

            const int ru = 2 * i;
            const int rv = ru + 1;

            // Radial terms
            T rpow = r2;  // r^(2*1)
            for (int j = 0; j < num_radial_; ++j) {
                A(ru, j) = intr[0] * x * rpow;
                A(rv, j) = intr[1] * y * rpow;
                rpow *= r2;
            }

            // Tangential terms
            const int idx_p1 = num_radial_;
            const int idx_p2 = num_radial_ + 1;
            A(ru, idx_p1) = intr[0] * (T(2.0) * x * y);
            A(ru, idx_p2) = intr[0] * (r2 + T(2.0) * x * x);
            A(rv, idx_p1) = intr[1] * (r2 + T(2.0) * y * y);
            A(rv, idx_p2) = intr[1] * (T(2.0) * x * y);

            b(ru) = du;
            b(rv) = dv;
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> alpha =
            (A.transpose() * A).ldlt().solve(A.transpose() * b);
        Eigen::Matrix<T, Eigen::Dynamic, 1> r = A * alpha - b;
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
    const std::vector<Observation>& obs,
    int num_radial,
    const Intrinsic& initial_guess,
    bool verb
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
