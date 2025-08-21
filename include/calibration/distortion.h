/** @brief Linear least squares design matrix for distortion parameters */

#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

namespace vitavision {

template<typename T>
struct Observation {
    T x, y;   // normalized undistorted coords
    T u, v;   // observed distorted pixel coords
};

Eigen::Vector2d apply_distortion(const Eigen::Vector2d& norm_xy,
                                 const Eigen::VectorXd& coeffs);

template<typename T>
struct DistortionWithResiduals {
    Eigen::Matrix<T, Eigen::Dynamic, 1> distortion;
    Eigen::Matrix<T, Eigen::Dynamic, 1> residuals;
};

template<typename T>
DistortionWithResiduals<T> fit_distortion_full(
    const std::vector<Observation<T>>& obs,
    T fx, T fy, T cx, T cy,
    int num_radial = 2
) {
    if (obs.size() < 8) {
        // Return empty result instead of throwing exception
        // This is safer with automatic differentiation
        Eigen::Matrix<T, Eigen::Dynamic, 1> empty_vec(1);
        empty_vec(0) = T(0);
        return {empty_vec, empty_vec};
    }

    const int M = num_radial + 2;  // radial + tangential coeffs
    const int N = static_cast<int>(obs.size());
    const int rows = N * 2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A(rows, M);
    Eigen::Matrix<T, Eigen::Dynamic, 1> b(rows);

    A.setZero();
    b.setZero();

    const int idx_p1 = num_radial;
    const int idx_p2 = num_radial + 1;

    for (int i = 0; i < N; ++i) {
        const T x = T(obs[i].x);
        const T y = T(obs[i].y);
        const T r2 = x * x + y * y;

        const T u0 = fx * x + cx;
        const T v0 = fy * y + cy;

        const T du = T(obs[i].u) - u0;
        const T dv = T(obs[i].v) - v0;

        const int ru = 2 * i;
        const int rv = ru + 1;

        // Radial terms
        T rpow = r2;  // r^(2*1)
        for (int j = 0; j < num_radial; ++j) {
            A(ru, j) = fx * x * rpow;
            A(rv, j) = fy * y * rpow;
            rpow *= r2;
        }

        // Tangential terms
        A(ru, idx_p1) = fx * (T(2.0) * x * y);
        A(ru, idx_p2) = fx * (r2 + T(2.0) * x * x);
        A(rv, idx_p1) = fy * (r2 + T(2.0) * y * y);
        A(rv, idx_p2) = fy * (T(2.0) * x * y);

        b(ru) = du;
        b(rv) = dv;
    }

    #if 0
    auto alpha = A.colPivHouseholderQr().solve(b);
    #elif 0
    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto alpha = svd.solve(b);
    #else
    Eigen::Matrix<T, Eigen::Dynamic, 1> alpha = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    #endif

    Eigen::Matrix<T, Eigen::Dynamic, 1> r = A * alpha - b;

    return {alpha, r};
}

template<typename T>
Eigen::VectorXd fit_distortion(
    const std::vector<Observation<T>>& obs,
    T fx, T fy, T cx, T cy,
    size_t num_radial = 2
) {
    return fit_distortion_full(obs, fx, fy, cx, cy, num_radial).distortion;
}

}  // namespace vitavision
