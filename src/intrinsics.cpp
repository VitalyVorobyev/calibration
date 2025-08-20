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

static Eigen::Matrix4d compute_covariance(
    size_t n_obs,
    const IntrinsicsVPResidual& vp,
    double const* intrinsics,
    ceres::Problem& problem
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
    const double rmse = std::sqrt(ssr / m);
    const double sigma2 = ssr / std::max(1, dof);
    std::cout << "RMSE (pixels): " << rmse << "\n";

    // Covariance (approximate): scale by residual variance.
    ceres::Covariance::Options cov_opts;
    ceres::Covariance cov(cov_opts);
    std::vector<std::pair<const double*, const double*>> blocks = { {intrinsics, intrinsics} };
    if (!cov.Compute(blocks, &problem)) {
        std::cerr << "Covariance computation failed.\n";
        return Eigen::Matrix4d::Zero();
    }

    double cov4x4[16];
    cov.GetCovarianceBlock(intrinsics, intrinsics, cov4x4);

    // Scale by estimated noise variance (unit weights assumption).
    Eigen::Map<Eigen::Matrix4d> covar(cov4x4);
    covar *= sigma2;
    return covar;
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
    result.covariance = compute_covariance(obs.size(), *costptr, intrinsics, problem);

    return result;
}

#if 0
// ---------- Demo / usage ----------
int main() {
  // Synthetic data for demonstration.
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> uni(-0.6, 0.6);
  std::normal_distribution<double> noise(0.0, 0.2); // pixel noise

  // True intrinsics & distortion
  const double fx_true = 800.0, fy_true = 820.0, cx_true = 640.0, cy_true = 360.0;
  const std::vector<double> k_true = {-0.20, 0.03}; // k1, k2
  const double p1_true = 0.0010, p2_true = -0.0005;

  const int N = 300;     // points
  const int K = 2;       // number of radial coeffs to estimate
  std::vector<Observation> obs; obs.reserve(N);

  auto distort_then_project = [&](double x, double y,
                                  double& u, double& v) {
    const double r2 = x*x + y*y;
    double radial = 1.0;
    double rpow = r2;
    for (double kj : k_true) { radial += kj * rpow; rpow *= r2; }

    double x_t = x * radial + 2.0 * p1_true * x * y + p2_true * (r2 + 2.0 * x * x);
    double y_t = y * radial + p1_true * (r2 + 2.0 * y * y) + 2.0 * p2_true * x * y;

    u = fx_true * x_t + cx_true;
    v = fy_true * y_t + cy_true;
  };

  for (int i = 0; i < N; ++i) {
    double x = uni(rng), y = uni(rng);
    double u, v; distort_then_project(x, y, u, v);
    obs.push_back({x, y, u + noise(rng), v + noise(rng)});
  }

  // Initial guess (slightly off)
  double intrinsics[4] = {780.0, 800.0, 630.0, 350.0};

  auto result = optimize_intrinsics(obs, K, intrinsics, true);

  std::cout << "Std. deviations (1-sigma):\n";
  for (int i = 0; i < 4; ++i) {
    double s = std::sqrt(std::max(0.0, result.covariance(i,i)));
    std::cout << "  param[" << i << "] = " << s << "\n";
  }
  return 0;
}
#endif

}  // namespace vitavision
