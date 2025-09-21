/**
 * @file distortion.h
 * @brief Lens distortion models and correction algorithms
 * @ingroup distortion_correction
 *
 * This file provides comprehensive lens distortion functionality including:
 * - C++20 concept-based distortion model interface
 * - Radial and tangential distortion correction
 * - Forward and inverse distortion mapping
 * - Linear least squares design matrix computation for distortion parameters
 */

#pragma once

// std
#include <algorithm>
#include <concepts>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/core/cameramatrix.h"

namespace calib {

/**
 * @brief Concept defining the interface for lens distortion models
 * @ingroup distortion_correction
 *
 * A distortion model must provide both forward distortion and inverse undistortion
 * operations for normalized 2D coordinates. This concept ensures type safety and
 * compile-time verification of distortion model implementations.
 *
 * @tparam D The distortion model type
 *
 * Requirements:
 * - Must have a Scalar type member
 * - Must provide distort() method for forward mapping
 * - Must provide undistort() method for inverse mapping
 */
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

/**
 * @brief Observation structure for distortion parameter estimation
 * @ingroup distortion_correction
 *
 * This structure holds corresponding normalized undistorted coordinates
 * and observed distorted pixel coordinates, used for distortion parameter
 * estimation through linear least squares or bundle adjustment.
 *
 * @tparam T Scalar type (float, double, etc.)
 */
template <typename T>
struct Observation final {
    T x, y;  ///< Normalized undistorted coordinates
    T u, v;  ///< Observed distorted pixel coordinates
};

/**
 * @brief Apply lens distortion to normalized coordinates
 * @ingroup distortion_correction
 *
 * Applies radial and tangential distortion to normalized 2D coordinates
 * using the Brown-Conrady distortion model. The distortion coefficients
 * are ordered as [k1, k2, ..., kn, p1, p2] where ki are radial distortion
 * coefficients and p1, p2 are tangential distortion coefficients.
 *
 * @tparam T Scalar type (float, double, etc.)
 * @param norm_xy Normalized undistorted 2D coordinates
 * @param coeffs Distortion coefficients vector (minimum 2 elements)
 * @return Distorted normalized coordinates
 * @throws std::runtime_error if coeffs has fewer than 2 elements
 *
 * @note The input coordinates should be normalized by the camera intrinsics
 */
template <typename T>
[[nodiscard]]
auto apply_distortion(const Eigen::Matrix<T, 2, 1>& norm_xy,
                      const Eigen::Matrix<T, Eigen::Dynamic, 1>& coeffs) -> Eigen::Matrix<T, 2, 1> {
    if (coeffs.size() < 2) {
        throw std::runtime_error("Insufficient distortion coefficients");
    }

    const int num_radial_coeffs = static_cast<int>(coeffs.size()) - 2;
    const T& x_coord = norm_xy.x();
    const T& y_coord = norm_xy.y();
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

    constexpr int k_num_undistort_iters = 5;
    Eigen::Matrix<T, 2, 1> undistorted_xy = norm_xy;
    for (int iter = 0; iter < k_num_undistort_iters; ++iter) {
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
    constexpr Scalar k_two = Scalar(2.0);
    for (int i = 0; i < grid; ++i) {
        Scalar x_coord =
            -lim + k_two * lim * static_cast<Scalar>(i) / static_cast<Scalar>(grid - 1);
        for (int j = 0; j < grid; ++j) {
            Scalar y_coord =
                -lim + k_two * lim * static_cast<Scalar>(j) / static_cast<Scalar>(grid - 1);
            Eigen::Matrix<Scalar, 2, 1> und(x_coord, y_coord);
            Eigen::Matrix<Scalar, 2, 1> dst = apply_distortion(und, forward);
            obs.push_back({dst.x(), dst.y(), x_coord, y_coord});
        }
    }

    auto inv_opt = fit_distortion_full(obs, {1.0, 1.0, 0.0, 0.0, 0.0}, num_radial);
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
auto fit_distortion_full(
    const std::vector<Observation<T>>& observations, const CameraMatrixT<T>& intrinsics,
    int num_radial = 2, std::span<const int> fixed_indices = {},
    std::span<const T> fixed_values = {}) -> std::optional<DistortionWithResiduals<T>> {
    constexpr int k_min_observations = 8;
    if (observations.size() < k_min_observations) {
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

    for (int obs_idx = 0; obs_idx < num_obs; ++obs_idx) {
        const T x_coord = T(observations[obs_idx].x);
        const T y_coord = T(observations[obs_idx].y);
        const T r2_val = x_coord * x_coord + y_coord * y_coord;

        const T undistorted_u = intrinsics.fx * x_coord + intrinsics.skew * y_coord + intrinsics.cx;
        const T undistorted_v = intrinsics.fy * y_coord + intrinsics.cy;

        const T residual_u = T(observations[obs_idx].u) - undistorted_u;
        const T residual_v = T(observations[obs_idx].v) - undistorted_v;

        const int row_u = 2 * obs_idx;
        const int row_v = row_u + 1;

        // Radial terms
        T rpow = r2_val;
        for (int j = 0; j < num_radial; ++j) {
            design_matrix(row_u, j) =
                intrinsics.fx * x_coord * rpow + intrinsics.skew * y_coord * rpow;
            design_matrix(row_v, j) = intrinsics.fy * y_coord * rpow;
            rpow *= r2_val;
        }

        // Tangential terms
        design_matrix(row_u, idx_p1) = intrinsics.fx * (T(2.0) * x_coord * y_coord) +
                                       intrinsics.skew * (r2_val + T(2.0) * y_coord * y_coord);
        design_matrix(row_u, idx_p2) = intrinsics.fx * (r2_val + T(2.0) * x_coord * x_coord) +
                                       intrinsics.skew * (T(2.0) * x_coord * y_coord);
        design_matrix(row_v, idx_p1) = intrinsics.fy * (r2_val + T(2.0) * y_coord * y_coord);
        design_matrix(row_v, idx_p2) = intrinsics.fy * (T(2.0) * x_coord * y_coord);

        rhs(row_u) = residual_u;
        rhs(row_v) = residual_v;
    }

    if (fixed_indices.empty()) {
        Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
            design_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        auto alpha = svd.solve(rhs);
        Eigen::Matrix<T, Eigen::Dynamic, 1> residuals = design_matrix * alpha - rhs;
        return DistortionWithResiduals<T>{alpha, residuals};
    }

    std::vector<std::pair<int, T>> fixed_pairs;
    fixed_pairs.reserve(fixed_indices.size());
    for (size_t i = 0; i < fixed_indices.size(); ++i) {
        int idx = fixed_indices[i];
        T value = T(0);
        if (i < fixed_values.size()) {
            value = fixed_values[i];
        }
        fixed_pairs.emplace_back(idx, value);
    }
    std::sort(fixed_pairs.begin(), fixed_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    fixed_pairs.erase(std::unique(fixed_pairs.begin(), fixed_pairs.end(),
                                  [](const auto& a, const auto& b) { return a.first == b.first; }),
                      fixed_pairs.end());

    std::vector<int> fixed_only;
    fixed_only.reserve(fixed_pairs.size());
    for (const auto& [idx, value] : fixed_pairs) {
        if (idx < 0 || idx >= num_coeffs) {
            throw std::invalid_argument("Fixed distortion index out of range");
        }
        fixed_only.push_back(idx);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> alpha =
        Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(num_coeffs);
    for (const auto& [idx, value] : fixed_pairs) {
        alpha(idx) = value;
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> rhs_adjusted = rhs;
    for (const auto& [idx, value] : fixed_pairs) {
        rhs_adjusted -= design_matrix.col(idx) * value;
    }

    std::vector<int> free_indices;
    free_indices.reserve(num_coeffs - static_cast<int>(fixed_pairs.size()));
    for (int idx = 0; idx < num_coeffs; ++idx) {
        if (!std::binary_search(fixed_only.begin(), fixed_only.end(), idx)) {
            free_indices.push_back(idx);
        }
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> residuals;
    if (free_indices.empty()) {
        residuals = design_matrix * alpha - rhs;
        return DistortionWithResiduals<T>{alpha, residuals};
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> free_design(
        design_matrix.rows(), static_cast<int>(free_indices.size()));
    for (int col = 0; col < static_cast<int>(free_indices.size()); ++col) {
        free_design.col(col) = design_matrix.col(free_indices[col]);
    }

    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
        free_design, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix<T, Eigen::Dynamic, 1> free_alpha = svd.solve(rhs_adjusted);

    for (int col = 0; col < static_cast<int>(free_indices.size()); ++col) {
        alpha(free_indices[col]) = free_alpha(col);
    }

    residuals = design_matrix * alpha - rhs;
    return DistortionWithResiduals<T>{std::move(alpha), std::move(residuals)};
}

template <typename T>
auto fit_distortion(const std::vector<Observation<T>>& observations,
                    const CameraMatrixT<T>& intrinsics, int num_radial = 2,
                    std::span<const int> fixed_indices = {}, std::span<const T> fixed_values = {})
    -> std::optional<DistortionWithResiduals<T>> {
    return fit_distortion_full(observations, intrinsics, num_radial, fixed_indices, fixed_values);
}

inline auto fit_distortion_dual(
    const std::vector<Observation<double>>& observations, const CameraMatrix& intrinsics,
    int num_radial = 2, std::span<const int> fixed_indices = {},
    std::span<const double> fixed_values = {}) -> std::optional<DualDistortionWithResiduals> {
    auto forward =
        fit_distortion_full(observations, intrinsics, num_radial, fixed_indices, fixed_values);
    if (!forward) {
        return std::nullopt;
    }

    std::vector<Observation<double>> inv_observations;
    inv_observations.reserve(observations.size());
    for (const auto& obs : observations) {
        double y_dist = (obs.v - intrinsics.cy) / intrinsics.fy;
        double x_dist = (obs.u - intrinsics.cx - intrinsics.skew * y_dist) / intrinsics.fx;
        double u_undist = intrinsics.fx * obs.x + intrinsics.skew * obs.y + intrinsics.cx;
        double v_undist = intrinsics.fy * obs.y + intrinsics.cy;
        inv_observations.push_back({x_dist, y_dist, u_undist, v_undist});
    }

    auto inverse =
        fit_distortion_full(inv_observations, intrinsics, num_radial, fixed_indices, fixed_values);
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
