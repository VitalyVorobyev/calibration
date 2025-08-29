/** @brief Linear least squares design matrix for distortion parameters */

#pragma once

// std
#include <optional>
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

namespace vitavision {

template<typename T>
struct Observation final {
    T x, y;   // normalized undistorted coords
    T u, v;   // observed distorted pixel coords
};

template<typename T>
Eigen::Matrix<T, 2, 1> apply_distortion(
    const Eigen::Matrix<T, 2, 1>& norm_xy,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& coeffs
) {
    if (coeffs.size() < 2) {
        throw std::runtime_error("Insufficient distortion coefficients");
    }

    const int num_k = static_cast<int>(coeffs.size()) - 2;
    const T x = norm_xy.x();
    const T y = norm_xy.y();
    T r2 = x * x + y * y;
    T radial = T(1);
    T rpow = r2;
    for (int i = 0; i < num_k; ++i) {
        radial += T(coeffs[i]) * rpow;
        rpow *= r2;
    }
    T p1 = T(coeffs[num_k]);
    T p2 = T(coeffs[num_k + 1]);
    T xt = x * radial + T(2) * p1 * x * y + p2 * (r2 + T(2) * x * x);
    T yt = y * radial + p1 * (r2 + T(2) * y * y) + T(2) * p2 * x * y;
    return {xt, yt};
}

template<typename T>
struct DistortionWithResiduals final {
    Eigen::Matrix<T, Eigen::Dynamic, 1> distortion;
    Eigen::Matrix<T, Eigen::Dynamic, 1> residuals;
};

struct DualDistortion final {
    Eigen::VectorXd forward;  ///< Coefficients for distortion
    Eigen::VectorXd inverse;  ///< Coefficients for undistortion

    template<typename T>
    Eigen::Matrix<T,2,1> distort(const Eigen::Matrix<T,2,1>& norm_xy) const {
        Eigen::Matrix<T,Eigen::Dynamic,1> coeffs = forward.template cast<T>();
        return apply_distortion(norm_xy, coeffs);
    }

    template<typename T>
    Eigen::Matrix<T,2,1> undistort(const Eigen::Matrix<T,2,1>& dist_xy) const {
        Eigen::Matrix<T,Eigen::Dynamic,1> coeffs = inverse.template cast<T>();
        return apply_distortion(dist_xy, coeffs);
    }
};

struct DualDistortionWithResiduals final {
    DualDistortion distortion;
    Eigen::VectorXd residuals;
};

template<typename T>
std::optional<DistortionWithResiduals<T>> fit_distortion_full(
    const std::vector<Observation<T>>& obs,
    T fx, T fy, T cx, T cy,
    int num_radial = 2
) {
    if (obs.size() < 8) {
        return std::nullopt;
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

    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto alpha = svd.solve(b);
    Eigen::Matrix<T, Eigen::Dynamic, 1> r = A * alpha - b;

    return DistortionWithResiduals<T>{alpha, r};
}

template<typename T>
std::optional<DistortionWithResiduals<T>> fit_distortion(
    const std::vector<Observation<T>>& obs,
    T fx, T fy, T cx, T cy,
    int num_radial = 2
) {
    return fit_distortion_full(obs, fx, fy, cx, cy, num_radial);
}

inline std::optional<DualDistortionWithResiduals> fit_distortion_dual(
    const std::vector<Observation<double>>& obs,
    double fx, double fy, double cx, double cy,
    int num_radial = 2
) {
    auto forward = fit_distortion_full(obs, fx, fy, cx, cy, num_radial);
    if (!forward) {
        return std::nullopt;
    }

    std::vector<Observation<double>> inv_obs;
    inv_obs.reserve(obs.size());
    for (const auto& o : obs) {
        double xd = (o.u - cx) / fx;
        double yd = (o.v - cy) / fy;
        double u_ud = fx * o.x + cx;
        double v_ud = fy * o.y + cy;
        inv_obs.push_back({xd, yd, u_ud, v_ud});
    }

    auto inverse = fit_distortion_full(inv_obs, fx, fy, cx, cy, num_radial);
    if (!inverse) {
        return std::nullopt;
    }

    DualDistortionWithResiduals out;
    out.distortion.forward = forward->distortion;
    out.distortion.inverse = inverse->distortion;
    out.residuals = forward->residuals;
    return out;
}

}  // namespace vitavision
