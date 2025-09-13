#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
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
//   std::optional<Model> fit(const std::vector<Datum>& data, std::span<const int> sample) const;
//   double residual(const Model& M, const Datum& d) const;
//   // optional:
//   std::optional<Model> refit(const std::vector<Datum>& data, std::span<const int> inliers) const;
//   bool is_degenerate(const std::vector<Datum>& data, std::span<const int> sample) const;
// };

namespace detail {
template <typename Estimator>
concept HasRefit =
    requires(const Estimator& est, const std::vector<typename Estimator::Datum>& data,
             std::span<const int> idxs) {
        { est.refit(data, idxs) } -> std::same_as<std::optional<typename Estimator::Model>>;
    };

template <typename Estimator>
concept HasDegeneracyCheck =
    requires(const Estimator& est, const std::vector<typename Estimator::Datum>& data,
             std::span<const int> idxs) {
        { est.is_degenerate(data, idxs) } -> std::same_as<bool>;
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
}  // namespace detail

template <class Estimator>
auto ransac(const std::vector<typename Estimator::Datum>& data, const Estimator& est,
            const RansacOptions& opts = {}) -> RansacResult<typename Estimator::Model> {
    // using Datum = typename Estimator::Datum;
    using Model = typename Estimator::Model;

    RansacResult<Model> best;
    if (data.size() < Estimator::k_min_samples) return best;

    std::mt19937_64 rng(opts.seed);
    std::uniform_int_distribution<int> uni(0, static_cast<int>(data.size()) - 1);

    // TODO: iota + shuffle
    auto sample_k_unique = [&](size_t k, std::vector<int>& out) {
        out.resize(k);
        // Simple rejection sampling (OK for small k, e.g., 4 for H)
        for (;;) {
            for (size_t i = 0; i < k; ++i) {
                out[i] = uni(rng);
            }
            std::sort(out.begin(), out.end());
            if (std::unique(out.begin(), out.end()) == out.end()) {
                return;
            }
        }
    };

    auto maybe_update_iters = [&](int iters_so_far, double inlier_ratio) {
        if (opts.confidence <= 0.0 || inlier_ratio <= 0.0) return opts.max_iters;
        // N >= log(1 - p) / log(1 - w^m)
        double p = opts.confidence;
        double w = inlier_ratio;
        double m = static_cast<double>(Estimator::k_min_samples);
        double denom = std::log(std::max(1e-12, 1.0 - std::pow(w, m)));
        if (denom >= 0.0) {
            return opts.max_iters;
        }  // avoid degenerate
        int N = static_cast<int>(std::ceil(std::log(1.0 - p) / denom));
        return std::clamp(N, iters_so_far, opts.max_iters);
    };

    int dynamic_max_iters = opts.max_iters;
    std::vector<int> idxs;
    std::vector<int> inliers;
    std::vector<double> inlier_residuals;

    for (int it = 0; it < dynamic_max_iters; ++it) {
        sample_k_unique(Estimator::k_min_samples, idxs);

        if constexpr (detail::HasDegeneracyCheck<Estimator>) {
            if (est.is_degenerate(data, std::span<const int>(idxs))) {
                continue;
            }
        }

        auto model_opt = est.fit(data, std::span<const int>(idxs));
        if (!model_opt) {
            continue;
        }
        const Model& M = *model_opt;

        inliers.clear();
        inlier_residuals.clear();
        inliers.reserve(data.size());
        inlier_residuals.reserve(data.size());

        for (int i = 0; i < static_cast<int>(data.size()); ++i) {
            double r = est.residual(M, data[i]);
            if (r <= opts.thresh) {
                inliers.push_back(i);
                inlier_residuals.push_back(r);
            }
        }

        if (static_cast<int>(inliers.size()) < opts.min_inliers) {
            continue;
        }

        // optional refit on inliers
        Model M_refit = M;
        if (opts.refit_on_inliers) {
            if constexpr (detail::HasRefit<Estimator>) {
                if (auto m2 = est.refit(data, std::span<const int>(inliers))) {
                    M_refit = *m2;
                    // recompute inliers + residuals after refit
                    inliers.clear();
                    inlier_residuals.clear();
                    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
                        double r = est.residual(M_refit, data[i]);
                        if (r <= opts.thresh) {
                            inliers.push_back(i);
                            inlier_residuals.push_back(r);
                        }
                    }
                }
            }
        }

        double rms_now = detail::rms(inlier_residuals);

        // choose best by: more inliers, then lower RMS
        bool better = false;
        if (!best.success) {
            better = true;
        } else if (inliers.size() > best.inliers.size()) {
            better = true;
        } else if (inliers.size() == best.inliers.size() && rms_now < best.inlier_rms) {
            better = true;
        }

        if (better) {
            best.success = true;
            best.model = M_refit;
            best.inliers = inliers;
            best.inlier_rms = rms_now;
            best.iters = it + 1;

            // Update dynamic iteration budget based on observed inlier ratio
            double w = static_cast<double>(best.inliers.size()) / static_cast<double>(data.size());
            dynamic_max_iters = maybe_update_iters(best.iters, w);

            // Early exit if perfect consensus
            if (best.inliers.size() == data.size()) {
                break;
            }
        }
    }

    return best;
}

}  // namespace calib
