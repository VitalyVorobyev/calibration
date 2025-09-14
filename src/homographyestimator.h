#pragma once

// std
#include <optional>
#include <span>
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/planarpose.h"  // for PlanarObservation

namespace calib {

struct HomographyEstimator final {
    using Datum = PlanarObservation;
    using Model = Eigen::Matrix3d;
    static constexpr size_t k_min_samples = 4;

    // --- Estimator API ---
    [[nodiscard]]
    static auto fit(const std::vector<Datum>& data,
                    std::span<const int> sample) -> std::optional<Model>;

    [[nodiscard]]
    static auto residual(const Model& hmtx, const Datum& observation) -> double;

    // Optional: better final model on all inliers
    [[nodiscard]]
    static auto refit(const std::vector<Datum>& data,
                      std::span<const int> inliers) -> std::optional<Model>;

    // Optional: reject degenerate minimal sets (near-collinear points)
    [[nodiscard]]
    static auto is_degenerate(const std::vector<Datum>& data, std::span<const int> sample) -> bool;
};

}  // namespace calib
