#pragma once

// std
#include <optional>
#include <span>
#include <vector>

// eigen
#include <Eigen/Core>

namespace calib {

/**
 * @brief Minimal plane estimator for RANSAC.
 *
 * The plane is represented by normalized coefficients [nx, ny, nz, d] such that
 * nx*x + ny*y + nz*z + d = 0 and \|n\| = 1.
 */
struct PlaneEstimator final {
    using Datum = Eigen::Vector3d;
    using Model = Eigen::Vector4d;
    static constexpr size_t k_min_samples = 3;  ///< Three points define a plane

    [[nodiscard]]
    static auto fit(const std::vector<Datum>& data, std::span<const int> sample)
        -> std::optional<Model>;

    [[nodiscard]]
    static auto residual(const Model& plane, const Datum& p) -> double;

    // Optional: refine model on a larger set of inliers
    [[nodiscard]]
    static auto refit(const std::vector<Datum>& data, std::span<const int> inliers)
        -> std::optional<Model>;

    // Optional: reject degenerate samples (collinear points)
    [[nodiscard]]
    static auto is_degenerate(const std::vector<Datum>& data, std::span<const int> sample) -> bool;
};

}  // namespace calib
