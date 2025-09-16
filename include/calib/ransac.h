#pragma once

#include <algorithm>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <limits>
#include <numeric>  // for std::iota
#include <optional>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

namespace calib {

struct RansacOptions final {
    int max_iters = 1000;
    double thresh = 2.0;       // residual threshold for inliers (units = estimator's residual)
    int min_inliers = 12;      // minimal consensus required to accept a model
    double confidence = 0.99;  // (optional) if >0, adaptively stop when inlier ratio is high
    uint64_t seed = 1234567;
    bool refit_on_inliers = true;
};

// Result is generic in the Model type.
template <class ModelT>
struct RansacResult final {
    bool success{false};
    ModelT model{};
    std::vector<int> inliers;
    double inlier_rms{std::numeric_limits<double>::infinity()};
    int iters{0};
};

// Estimator concept (informal; enforced via requires in use):
// struct Estimator {
//   using Datum = ...;
//   using Model = ...;
//   static constexpr size_t k_min_samples = ...;
//   static std::optional<Model> fit(const std::vector<Datum>& data, std::span<const int> sample);
//   static double residual(const Model& M, const Datum& d);
//   // optional:
//   static std::optional<Model> refit(const std::vector<Datum>& data, std::span<const int>
//   inliers); static bool is_degenerate(const std::vector<Datum>& data, std::span<const int>
//   sample);
// };

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
        spdlog::warn("RMS of empty set");
        return std::numeric_limits<double>::infinity();
    }
    double ss = std::accumulate(vals.begin(), vals.end(), 0.0,
                                [](double acc, double v) { return acc + v * v; });
    return std::sqrt(ss / static_cast<double>(vals.size()));
}

// Sample k unique indices from [0, size-1]
template <typename RNG>
inline void sample_k_unique(size_t k, size_t size, RNG& rng, std::vector<int>& out) {
    if (k > size) {
        spdlog::warn("sample_k_unique called with k > size");
        k = size;
    }

    // Create vector with indices 0, 1, 2, ..., size-1
    out.resize(size);
    std::iota(out.begin(), out.end(), 0);

    // Shuffle the vector
    std::shuffle(out.begin(), out.end(), rng);

    // Resize to keep only the first k elements
    out.resize(k);
}

// Calculate number of iterations based on inlier ratio
inline int calculate_iterations(double confidence, double inlier_ratio, int min_samples,
                                int iters_so_far, int max_iters) {
    if (confidence <= 0.0 || inlier_ratio <= 0.0) {
        return max_iters;
    }

    // N >= log(1 - p) / log(1 - w^m)
    double p = confidence;
    double w = inlier_ratio;
    auto m = static_cast<double>(min_samples);
    double denom = std::log(std::max(1e-12, 1.0 - std::pow(w, m)));

    if (denom >= 0.0) {
        return max_iters;  // avoid degenerate case
    }

    const int niter = static_cast<int>(std::ceil(std::log(1.0 - p) / denom));
    return std::clamp(niter, iters_so_far, max_iters);
}

// Find inliers for a given model
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

// Refit model on inliers if possible
template <typename Estimator, typename Model>
inline Model refit_model(const std::vector<typename Estimator::Datum>& data, const Model& model,
                         const std::vector<int>& inliers, double threshold,
                         std::vector<int>& updated_inliers,
                         std::vector<double>& updated_residuals) {
    Model refined_model = model;

    if constexpr (HasRefit<Estimator>) {
        if (auto m2 = Estimator::refit(data, std::span<const int>(inliers))) {
            refined_model = *m2;
            // Recompute inliers after refit
            find_inliers<Estimator>(data, refined_model, threshold, updated_inliers,
                                    updated_residuals);
        }
    }

    return refined_model;
}

// Check if new model is better than current best
inline bool is_better_model(bool has_current_best, size_t new_inlier_count, double new_inlier_rms,
                            size_t best_inlier_count, double best_inlier_rms) {
    return !has_current_best || (new_inlier_count > best_inlier_count) ||
           (new_inlier_count == best_inlier_count && new_inlier_rms < best_inlier_rms);
}

}  // namespace detail

template <class Estimator>
auto ransac(const std::vector<typename Estimator::Datum>& data,
            const RansacOptions& opts = {}) -> RansacResult<typename Estimator::Model> {
    using Model = typename Estimator::Model;

    RansacResult<Model> best;
    if (data.size() < Estimator::k_min_samples) {
        return best;
    }

    std::mt19937_64 rng(opts.seed);

    int dynamic_max_iters = opts.max_iters;
    std::vector<int> idxs;
    std::vector<int> inliers;
    std::vector<double> inlier_residuals;
    std::vector<int> refined_inliers;
    std::vector<double> refined_residuals;

    for (int it = 0; it < dynamic_max_iters; ++it) {
        detail::sample_k_unique(Estimator::k_min_samples, data.size(), rng, idxs);

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

        // Refit on inliers if requested
        Model model_refit = model;
        if (opts.refit_on_inliers) {
            refined_inliers = inliers;
            refined_residuals = inlier_residuals;
            model_refit = detail::refit_model<Estimator>(data, model, inliers, opts.thresh,
                                                         refined_inliers, refined_residuals);
            inliers = refined_inliers;
            inlier_residuals = refined_residuals;
        }

        double rms_now = detail::rms(inlier_residuals);

        bool better = detail::is_better_model(best.success, inliers.size(), rms_now,
                                              best.inliers.size(), best.inlier_rms);

        if (better) {
            best.success = true;
            best.model = model_refit;
            best.inliers = inliers;
            best.inlier_rms = rms_now;
            best.iters = it + 1;

            // Update dynamic iteration budget based on observed inlier ratio
            double w = static_cast<double>(best.inliers.size()) / static_cast<double>(data.size());
            dynamic_max_iters = detail::calculate_iterations(
                opts.confidence, w, Estimator::k_min_samples, best.iters, opts.max_iters);

            // Early exit if perfect consensus
            if (best.inliers.size() == data.size()) {
                break;
            }
        }
    }

    return best;
}

}  // namespace calib
