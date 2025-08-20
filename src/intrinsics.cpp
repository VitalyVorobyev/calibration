#include "calibration/intrinsics.h"

// std
#include <iostream>
#include <random>
#include <cmath>

// ceres
#include <ceres/ceres.h>

namespace vitavision {

// CostFunction that does variable projection and finite-difference Jacobians.
class IntrinsicsVPResidual : public ceres::CostFunction {
    void computeResiduals(const double intr[4], double* residuals) const {
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        LSDesign::build(obs_, num_radial_, intr[0], intr[1], intr[2], intr[3], A, b);
        Eigen::VectorXd alpha = LSDesign::solveNormal(A, b);

        // Predicted minus observed = (u0 + A*alpha) - (u_obs, v_obs)  ==  A*alpha - b
        Eigen::VectorXd r = A * alpha - b;
        // Copy out
        for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
    }

    std::vector<Observation> obs_;
    int num_radial_;

public:
    IntrinsicsVPResidual(std::vector<Observation> obs, int num_radial)
        : obs_(std::move(obs)), num_radial_(num_radial)
    {
        set_num_residuals(static_cast<int>(obs_.size()) * 2);
        // Single parameter block: [fx, fy, cx, cy]
        mutable_parameter_block_sizes()->push_back(4);
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        const double* intr = parameters[0];
        computeResiduals(intr, residuals);

        if (jacobians && jacobians[0]) {
            // Central finite differences on intrinsics
            double* J = jacobians[0];
            const int m = num_residuals();
            const double base[4] = {intr[0], intr[1], intr[2], intr[3]};

            std::vector<double> r_plus(m), r_minus(m);
            for (int k = 0; k < 4; ++k) {
                double step = 1e-6 * std::max(1.0, std::abs(base[k]));
                double intr_p[4] = { base[0], base[1], base[2], base[3] };
                double intr_m[4] = { base[0], base[1], base[2], base[3] };
                intr_p[k] += step;
                intr_m[k] -= step;

                computeResiduals(intr_p, r_plus.data());
                computeResiduals(intr_m, r_minus.data());

                for (int i = 0; i < m; ++i) {
                    J[i * 4 + k] = (r_plus[i] - r_minus[i]) / (2.0 * step);
                }
            }
        }
        return true;
    }

    // Compute optimal distortion coeffs for given intrinsics (useful after Solve).
    Eigen::VectorXd SolveDistortionFor(const double intr[4]) const {
        return fit_distortion(obs_, intr[0], intr[1], intr[2], intr[3], num_radial_);
    }
};

static bool compute_covariance(
    size_t n_obs,
    const IntrinsicsVPResidual& vp,
    double const* intrinsics,
    ceres::Problem& problem,
    IntrinsicOptimizationResult& result
) {
    // Recompute residuals and RMSE
    const int m = static_cast<int>(n_obs) * 2;
    std::vector<double> residuals(2 * n_obs);
    
    // Fix: Create parameter array for Evaluate
    const double* params[1] = {intrinsics};
    vp.Evaluate(params, residuals.data(), nullptr);

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
    #if 0
    auto costptr = std::make_unique<IntrinsicsVPResidual>(obs, num_radial);
    #else
    auto costptr = new IntrinsicsVPResidual(obs, num_radial);
    #endif

    problem.AddResidualBlock(costptr, /*loss=*/nullptr, intrinsics);

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
    
    // Fix: Pass the cost function object reference instead of intrinsics array
    if (!compute_covariance(obs.size(), *costptr, intrinsics, problem, result)) {
        std::cerr << "Covariance computation failed.\n";
    }

    return result;
}

}  // namespace vitavision
