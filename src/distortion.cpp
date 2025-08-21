#include "calibration/distortion.h"

namespace vitavision {

#if 0
struct LSDesign {
    // Build A * alpha ≈ b, where alpha = [k1..kK, p1, p2]^T
    static void build(const std::vector<Observation>& obs,
                      int num_radial,
                      double fx, double fy, double cx, double cy,
                      Eigen::MatrixXd& A, Eigen::VectorXd& b);

    // Solve (A^T A) alpha = A^T b
    static Eigen::VectorXd solveNormal(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
};

void LSDesign::build(const std::vector<Observation>& obs,
                     int num_radial,
                     double fx, double fy, double cx, double cy,
                     Eigen::MatrixXd& A, Eigen::VectorXd& b) {
    const int M = num_radial + 2;  // radial Ks + (p1, p2)
    const int N = static_cast<int>(obs.size());
    const int rows = N * 2;
    A.setZero(rows, M);
    b.setZero(rows);

    // Tangential p1, p2 (last two columns)
    const int idx_p1 = num_radial;
    const int idx_p2 = num_radial + 1;

    for (int i = 0; i < N; ++i) {
        const double x = obs[i].x;
        const double y = obs[i].y;
        const double r2 = x*x + y*y;

        const double u0 = fx * x + cx;
        const double v0 = fy * y + cy;

        const double du = obs[i].u - u0;
        const double dv = obs[i].v - v0;

        const int ru = 2*i;
        const int rv = 2*i + 1;

        // Radial columns
        double rpow = r2; // r^(2*1)
        for (int j = 0; j < num_radial; ++j) {
            A(ru, j) = fx * x * rpow;
            A(rv, j) = fy * y * rpow;
            rpow *= r2; // next power
        }

        // u_tangential = fx*(2*p1*x*y + p2*(r^2 + 2x^2))
        A(ru, idx_p1) = fx * (2.0 * x * y);
        A(ru, idx_p2) = fx * (r2 + 2.0 * x * x);

        // v_tangential = fy*(p1*(r^2 + 2y^2) + 2*p2*x*y)
        A(rv, idx_p1) = fy * (r2 + 2.0 * y * y);
        A(rv, idx_p2) = fy * (2.0 * x * y);

        // Right-hand side: observed - undistorted
        b(ru) = du;
        b(rv) = dv;
    }
}

// Solve (A^T A) alpha = A^T b
Eigen::VectorXd LSDesign::solveNormal(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    // Use more robust QR decomposition instead of the normal equations
    // This avoids potential numerical issues with LDLT
    return A.colPivHouseholderQr().solve(b);
}
#endif

Eigen::Vector2d apply_distortion(
    const Eigen::Vector2d& norm_xy,
    const Eigen::VectorXd& coeffs
) {
    if (coeffs.size() < 2) {
        throw std::runtime_error("Insufficient distortion coefficients");
    }

    const double x = norm_xy.x();
    const double y = norm_xy.y();
    const double r2 = x*x + y*y;

    const int num_k_radial = coeffs.size() - 2;
    const double p1 = coeffs[num_k_radial];
    const double p2 = coeffs[num_k_radial + 1];

    // Apply radial distortion
    double radial = 1.0;
    double rpow = r2;

    for (int i = 0; i < num_k_radial; ++i) {
        radial += coeffs[i] * rpow;
        rpow *= r2;
    }

    // Apply tangential distortion
    double x_t = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    double y_t = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    return { x_t, y_t };
}

#if 0
Eigen::VectorXd fit_distortion(
    const std::vector<Observation>& obs,
    double fx, double fy, double cx, double cy,
    size_t num_radial
) {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    LSDesign::build(obs, num_radial, fx, fy, cx, cy, A, b);
    return LSDesign::solveNormal(A, b);
}
#endif

#if 0
DistortionWithResiduals fit_distortion_full(
    const std::vector<Observation>& obs,
    double fx, double fy, double cx, double cy,
    size_t num_radial
) {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    LSDesign::build(obs, num_radial, fx, fy, cx, cy, A, b);
    auto alpha = LSDesign::solveNormal(A, b);
    auto resid = A * alpha - b;
    return { alpha, resid };
}
#endif

}  // namespace vitavision
