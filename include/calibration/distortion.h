/** @brief Linear least squares design matrix for distortion parameters */

#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

namespace vitavision {

struct Observation {
    double x, y;   // normalized undistorted coords
    double u, v;   // observed distorted pixel coords
};

Eigen::VectorXd fit_distortion(
    const std::vector<Observation>& obs,
    double fx, double fy, double cx, double cy,
    size_t num_radial = 2);

// TODO: make it implementation detail in cpp
struct LSDesign {
    // Build A * alpha ≈ b, where alpha = [k1..kK, p1, p2]^T
    static void build(const std::vector<Observation>& obs,
                      int num_radial,
                      double fx, double fy, double cx, double cy,
                      Eigen::MatrixXd& A, Eigen::VectorXd& b);

    // Solve (A^T A) alpha = A^T b
    static Eigen::VectorXd solveNormal(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
};

}  // namespace vitavision
