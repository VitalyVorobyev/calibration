#include "calibration/distortion.h"

namespace vitavision {

void LSDesign::build(const std::vector<Observation>& obs,
                     int num_radial,
                     double fx, double fy, double cx, double cy,
                     Eigen::MatrixXd& A, Eigen::VectorXd& b) {
    const int M = num_radial + 2;  // radial Ks + (p1, p2)
    const int rows = static_cast<int>(obs.size()) * 2;
    A.setZero(rows, M);
    b.setZero(rows);

    // Tangential p1, p2 (last two columns)
    const int idx_p1 = num_radial;
    const int idx_p2 = num_radial + 1;

    for (int i = 0, n = static_cast<int>(obs.size()); i < n; ++i) {
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
    return (A.transpose() * A).ldlt().solve(A.transpose() * b);
}

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

}  // namespace vitavision
