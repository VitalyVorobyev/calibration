#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "calib/io/serialization.h"  // to_json/from_json

namespace calib {

struct RansacOptions final {
    int max_iters = 1000;
    double thresh = 2.0;
    int min_inliers = 12;
    double confidence = 0.99;
    uint64_t seed = 1234567;
    bool refit_on_inliers = true;
};

template <class ModelT>
struct RansacResult final {
    bool success{false};
    ModelT model{};
    std::vector<int> inliers;
    double inlier_rms{std::numeric_limits<double>::infinity()};
    int iters{0};
};

namespace detail {

template <typename Estimator>
concept HasRefit =
    requires(const std::vector<typename Estimator::Datum>& data, std::span<const int> idxs) {
        { Estimator::refit(data, idxs) } -> std::same_as<std::optional<typename Estimator::Model>>;
    };

template <typename Estimator>
concept HasDegeneracyCheck =
    requires(const std::vector<typename Estimator::Datum>& data, std::span<const int> idxs) {
        { Estimator::is_degenerate(data, idxs) } -> std::same_as<bool>;
    };

inline auto rms(const std::vector<double>& vals) -> double {
    if (vals.empty()) {
        std::cout << "Warning: RMS of empty set\n";
        return std::numeric_limits<double>::infinity();
    }
    double ss = std::accumulate(vals.begin(), vals.end(), 0.0,
                                [](double acc, double v) { return acc + v * v; });
    return std::sqrt(ss / static_cast<double>(vals.size()));
}

inline int calculate_iterations(double confidence, double inlier_ratio, int min_samples,
                                int iters_so_far, int max_iters) {
    if (confidence <= 0.0 || inlier_ratio <= 0.0) {
        return max_iters;
    }
    double p = confidence;
    double w = inlier_ratio;
    auto m = static_cast<double>(min_samples);
    double denom = std::log(std::max(1e-12, 1.0 - std::pow(w, m)));
    if (denom >= 0.0) {
        return max_iters;
    }
    const int niter = static_cast<int>(std::ceil(std::log(1.0 - p) / denom));
    return std::clamp(niter, iters_so_far, max_iters);
}

template <typename Estimator, typename Model>
inline void find_inliers(const std::vector<typename Estimator::Datum>& data, const Model& model,
                         double threshold, std::vector<int>& inliers,
                         std::vector<double>& inlier_residuals) {
    inliers.clear();
    inlier_residuals.clear();
    inliers.reserve(data.size());
    inlier_residuals.reserve(data.size());
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        double r = Estimator::residual(model, data[i]);
        if (r <= threshold) {
            inliers.push_back(i);
            inlier_residuals.push_back(r);
        }
    }
}

template <typename Estimator, typename Model>
inline auto refit_model(const std::vector<typename Estimator::Datum>& data, const Model& model,
                        const std::vector<int>& inliers, double threshold,
                        std::vector<int>& updated_inliers, std::vector<double>& updated_residuals)
    -> Model {
    Model refined_model = model;
    if constexpr (HasRefit<Estimator>) {
        if (auto m2 = Estimator::refit(data, std::span<const int>(inliers))) {
            refined_model = *m2;
            find_inliers<Estimator>(data, refined_model, threshold, updated_inliers,
                                    updated_residuals);
        }
    }
    return refined_model;
}

inline auto is_better_model(bool has_current_best, size_t new_inlier_count, double new_inlier_rms,
                            size_t best_inlier_count, double best_inlier_rms) -> bool {
    return !has_current_best || (new_inlier_count > best_inlier_count) ||
           (new_inlier_count == best_inlier_count && new_inlier_rms < best_inlier_rms);
}

}  // namespace detail

template <class Estimator>
auto ransac(const std::vector<typename Estimator::Datum>& data, const RansacOptions& opts = {})
    -> RansacResult<typename Estimator::Model> {
    using Model = typename Estimator::Model;

    RansacResult<Model> best;
    if (data.size() < Estimator::k_min_samples) {
        return best;
    }

    std::vector<int> all_indices(data.size());
    std::vector<int> idxs(Estimator::k_min_samples);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    std::mt19937_64 rng(opts.seed);

    int dynamic_max_iters = opts.max_iters;
    std::vector<int> inliers;
    std::vector<double> inlier_residuals;
    std::vector<int> refined_inliers;
    std::vector<double> refined_residuals;

    for (int it = 0; it < dynamic_max_iters; ++it) {
        std::sample(all_indices.begin(), all_indices.end(), idxs.begin(), Estimator::k_min_samples,
                    rng);

        if constexpr (detail::HasDegeneracyCheck<Estimator>) {
            if (Estimator::is_degenerate(data, std::span<const int>(idxs))) {
                continue;
            }
        }

        auto model_opt = Estimator::fit(data, std::span<const int>(idxs));
        if (!model_opt) {
            continue;
        }
        const Model& model = *model_opt;

        detail::find_inliers<Estimator>(data, model, opts.thresh, inliers, inlier_residuals);

        if (static_cast<int>(inliers.size()) < opts.min_inliers) {
            continue;
        }

        Model model_refit = model;
        if (opts.refit_on_inliers) {
            refined_inliers = inliers;
            refined_residuals = inlier_residuals;
            model_refit = detail::refit_model<Estimator>(data, model, inliers, opts.thresh,
                                                         refined_inliers, refined_residuals);
        }

        const auto& final_inliers = opts.refit_on_inliers ? refined_inliers : inliers;
        const auto& final_residuals = opts.refit_on_inliers ? refined_residuals : inlier_residuals;
        double final_rms = detail::rms(final_residuals);

        if (detail::is_better_model(best.success, final_inliers.size(), final_rms,
                                    best.inliers.size(), best.inlier_rms)) {
            best.success = true;
            best.model = model_refit;
            best.inliers = final_inliers;
            best.inlier_rms = final_rms;
            best.iters = it + 1;
        }

        const double inlier_ratio =
            static_cast<double>(final_inliers.size()) / static_cast<double>(data.size());
        dynamic_max_iters = detail::calculate_iterations(opts.confidence, inlier_ratio,
                                                         static_cast<int>(Estimator::k_min_samples),
                                                         it + 1, opts.max_iters);
    }

    return best;
}

}  // namespace calib
