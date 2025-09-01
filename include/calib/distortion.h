/** @brief Linear least squares design matrix for distortion parameters */

#pragma once

// std
#include <concepts>
#include <optional>
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

namespace calib {

template <typename D>
concept distortion_model =
    requires(const D& distortion, const Eigen::Matrix<typename D::Scalar, 2, 1>& point2d) {
        {
            distortion.template distort<typename D::Scalar>(point2d)
        } -> std::same_as<Eigen::Matrix<typename D::Scalar, 2, 1>>;
        {
            distortion.template undistort<typename D::Scalar>(point2d)
        } -> std::same_as<Eigen::Matrix<typename D::Scalar, 2, 1>>;
    };

template <typename T>
struct Observation final {
    T x, y;  // normalized undistorted coords
    T u, v;  // observed distorted pixel coords
};

template <typename T>
[[nodiscard]]
auto apply_distortion(const Eigen::Matrix<T, 2, 1>& norm_xy,
                      const Eigen::Matrix<T, Eigen::Dynamic, 1>& coeffs) -> Eigen::Matrix<T, 2, 1> {
    if (coeffs.size() < 2) {
        throw std::runtime_error("Insufficient distortion coefficients");
    }

    const int num_radial_coeffs = static_cast<int>(coeffs.size()) - 2;
    const T x_coord = norm_xy.x();
    const T y_coord = norm_xy.y();
    T r2_val = x_coord * x_coord + y_coord * y_coord;
    T radial = T(1);
    T rpow = r2_val;
    for (int i = 0; i < num_radial_coeffs; ++i) {
        radial += T(coeffs[i]) * rpow;
        rpow *= r2_val;
    }
    T tangential1 = T(coeffs[num_radial_coeffs]);
    T tangential2 = T(coeffs[num_radial_coeffs + 1]);
    T x_distorted = x_coord * radial + T(2) * tangential1 * x_coord * y_coord +
                    tangential2 * (r2_val + T(2) * x_coord * x_coord);
    T y_distorted = y_coord * radial + tangential1 * (r2_val + T(2) * y_coord * y_coord) +
                    T(2) * tangential2 * x_coord * y_coord;
    return {x_distorted, y_distorted};
}

// Normalize and undistort pixel coordinates
template <typename T>
[[nodiscard]]
auto undistort(Eigen::Matrix<T, 2, 1> norm_xy,
               const Eigen::Matrix<T, Eigen::Dynamic, 1>& coeffs) -> Eigen::Matrix<T, 2, 1> {
    if (coeffs.size() < 2) {
        throw std::runtime_error("Insufficient distortion coefficients");
    }

    constexpr int kNumUndistortIters = 5;
    Eigen::Matrix<T, 2, 1> undistorted_xy = norm_xy;
    for (int iter = 0; iter < kNumUndistortIters; ++iter) {
        Eigen::Matrix<T, 2, 1> distorted_xy = apply_distortion(undistorted_xy, coeffs);
        undistorted_xy += norm_xy - distorted_xy;
    }
    return undistorted_xy;
}

template <typename T>
struct DistortionWithResiduals final {
    Eigen::Matrix<T, Eigen::Dynamic, 1> distortion;
    Eigen::Matrix<T, Eigen::Dynamic, 1> residuals;
};

template <typename Scalar_>
struct BrownConrady final {
    using Scalar = Scalar_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> coeffs;

    BrownConrady() = default;
    template <typename Derived>
    explicit BrownConrady(const Eigen::MatrixBase<Derived>& coeffs_in) : coeffs(coeffs_in) {}

    template <typename T>
    [[nodiscard]]
    auto distort(const Eigen::Matrix<T, 2, 1>& norm_xy) const -> Eigen::Matrix<T, 2, 1> {
        return apply_distortion(norm_xy, coeffs.template cast<T>().eval());
    }

    template <typename T>
    [[nodiscard]]
    auto undistort(const Eigen::Matrix<T, 2, 1>& distorted_xy) const -> Eigen::Matrix<T, 2, 1> {
        return calib::undistort(distorted_xy, coeffs.template cast<T>().eval());
    }
};
using BrownConradyd = BrownConrady<double>;

template <typename Scalar>
inline auto invert_brown_conrady(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& forward)
    -> Eigen::Matrix<Scalar, Eigen::Dynamic, 1> {
    if (forward.size() < 2) {
        throw std::runtime_error("Insufficient distortion coefficients");
    }
    int num_radial = static_cast<int>(forward.size()) - 2;

    constexpr int grid = 21;
    constexpr double lim = 1.0;
    std::vector<Observation<Scalar>> obs;
    obs.reserve(grid * grid);
    for (int i = 0; i < grid; ++i) {
        Scalar x_coord = -lim + 2.0 * lim * static_cast<Scalar>(i) / static_cast<Scalar>(grid - 1);
        for (int j = 0; j < grid; ++j) {
            Scalar y_coord =
                -lim + 2.0 * lim * static_cast<Scalar>(j) / static_cast<Scalar>(grid - 1);
            Eigen::Matrix<Scalar, 2, 1> und(x_coord, y_coord);
            Eigen::Matrix<Scalar, 2, 1> dst = apply_distortion(und, forward);
            obs.push_back({dst.x(), dst.y(), x_coord, y_coord});
        }
    }

    auto inv_opt = fit_distortion_full(obs, 1.0, 1.0, 0.0, 0.0, 0.0, num_radial);
    if (inv_opt.has_value()) {
        return inv_opt->distortion;
    }
    return Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(forward.size());
}

template <typename Scalar_>
struct DualBrownConrady final {
    using Scalar = Scalar_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> forward;  ///< Coefficients for distortion
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> inverse;  ///< Coefficients for undistortion

    DualBrownConrady() = default;

    explicit DualBrownConrady(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& coeffs)
        : forward(coeffs), inverse(invert_brown_conrady(coeffs)) {}

    template <typename T>
    [[nodiscard]]
    auto distort(const Eigen::Matrix<T, 2, 1>& norm_xy) const -> Eigen::Matrix<T, 2, 1> {
        return apply_distortion(norm_xy, forward.template cast<T>().eval());
    }

    template <typename T>
    [[nodiscard]]
    auto undistort(const Eigen::Matrix<T, 2, 1>& distorted_xy) const -> Eigen::Matrix<T, 2, 1> {
        return apply_distortion(distorted_xy, inverse.template cast<T>().eval());
    }
};

using DualDistortion = DualBrownConrady<double>;

struct DualDistortionWithResiduals final {
    DualDistortion distortion;
    Eigen::VectorXd residuals;
};

// TODO: refactor for camera_model as a template parameter
template <typename T>
[[nodiscard]]
auto fit_distortion_full(const std::vector<Observation<T>>& observations, T focal_x, T focal_y,
                         T center_x, T center_y, T skew_param,
                         int num_radial = 2) -> std::optional<DistortionWithResiduals<T>> {
    if (observations.size() < 8) {
        return std::nullopt;
    }

    const int num_coeffs = num_radial + 2;  // radial + tangential coeffs
    const int num_obs = static_cast<int>(observations.size());
    const int num_rows = num_obs * 2;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> design_matrix(num_rows, num_coeffs);
    Eigen::Matrix<T, Eigen::Dynamic, 1> rhs(num_rows);

    design_matrix.setZero();
    rhs.setZero();

    const int idx_p1 = num_radial;
    const int idx_p2 = num_radial + 1;

    for (int i = 0; i < num_obs; ++i) {
        const T x_coord = T(observations[i].x);
        const T y_coord = T(observations[i].y);
        const T r2_val = x_coord * x_coord + y_coord * y_coord;

        const T u0 = focal_x * x_coord + skew_param * y_coord + center_x;
        const T v0 = focal_y * y_coord + center_y;

        const T du = T(observations[i].u) - u0;
        const T dv = T(observations[i].v) - v0;

        const int ru = 2 * i;
        const int rv = ru + 1;

        // Radial terms
        T rpow = r2_val;  // r^(2*1)
        for (int j = 0; j < num_radial; ++j) {
            design_matrix(ru, j) = focal_x * x_coord * rpow + skew_param * y_coord * rpow;
            design_matrix(rv, j) = focal_y * y_coord * rpow;
            rpow *= r2_val;
        }

        // Tangential terms
        design_matrix(ru, idx_p1) = focal_x * (T(2.0) * x_coord * y_coord) +
                                    skew_param * (r2_val + T(2.0) * y_coord * y_coord);
        design_matrix(ru, idx_p2) = focal_x * (r2_val + T(2.0) * x_coord * x_coord) +
                                    skew_param * (T(2.0) * x_coord * y_coord);
        design_matrix(rv, idx_p1) = focal_y * (r2_val + T(2.0) * y_coord * y_coord);
        design_matrix(rv, idx_p2) = focal_y * (T(2.0) * x_coord * y_coord);

        rhs(ru) = du;
        rhs(rv) = dv;
    }

    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
        design_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto alpha = svd.solve(rhs);
    Eigen::Matrix<T, Eigen::Dynamic, 1> residuals = design_matrix * alpha - rhs;

    return DistortionWithResiduals<T>{alpha, residuals};
}

template <typename T>
std::optional<DistortionWithResiduals<T>> fit_distortion(const std::vector<Observation<T>>& obs,
                                                         T fx, T fy, T cx, T cy, T skew,
                                                         int num_radial = 2) {
    return fit_distortion_full(obs, fx, fy, cx, cy, skew, num_radial);
}

inline auto fit_distortion_dual(const std::vector<Observation<double>>& observations,
                                double focal_x, double focal_y, double center_x, double center_y,
                                double skew_param,
                                int num_radial = 2) -> std::optional<DualDistortionWithResiduals> {
    auto forward = fit_distortion_full(observations, focal_x, focal_y, center_x, center_y,
                                       skew_param, num_radial);
    if (!forward) {
        return std::nullopt;
    }

    std::vector<Observation<double>> inv_observations;
    inv_observations.reserve(observations.size());
    for (const auto& obs : observations) {
        double y_dist = (obs.v - center_y) / focal_y;
        double x_dist = (obs.u - center_x - skew_param * y_dist) / focal_x;
        double u_undist = focal_x * obs.x + skew_param * obs.y + center_x;
        double v_undist = focal_y * obs.y + center_y;
        inv_observations.push_back({x_dist, y_dist, u_undist, v_undist});
    }

    auto inverse = fit_distortion_full(inv_observations, focal_x, focal_y, center_x, center_y,
                                       skew_param, num_radial);
    if (!inverse) {
        return std::nullopt;
    }

    DualDistortionWithResiduals out;
    out.distortion.forward = forward->distortion;
    out.distortion.inverse = inverse->distortion;
    out.residuals = forward->residuals;
    return out;
}

}  // namespace calib
