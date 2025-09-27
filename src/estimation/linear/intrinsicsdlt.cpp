#include <calib/estimation/common/intrinsics_utils.h>
#include "calib/estimation/intrinsics.h"

// std
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <utility>

#include "calib/estimation/posefromhomography.h"  // for pose_from_homography
#include "calib/estimation/ransac.h"
#include "detail/homographyestimator.h"
#include "detail/zhang.h"  // for zhang_intrinsics_from_hs

namespace calib {

static auto symmetric_rms_px(const Eigen::Matrix3d& hmtx, const PlanarView& view,
                             std::span<const int> inliers) -> double {
    if (inliers.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const double sum_err = std::accumulate(
        inliers.begin(), inliers.end(), 0.0,
        [&](double acc, int idx) { return acc + HomographyEstimator::residual(hmtx, view[idx]); });
    return std::sqrt(sum_err / (2.0 * static_cast<double>(inliers.size())));
}

static auto compute_planar_homographies(const std::vector<PlanarView>& views,
                                        const std::optional<RansacOptions>& ransac_opts)
    -> std::vector<HomographyResult> {
    std::vector<HomographyResult> homographies;
    homographies.reserve(views.size());

    for (const auto& view : views) {
        HomographyResult result;

        if (view.size() < HomographyEstimator::k_min_samples) {
            result.success = false;
            homographies.push_back(std::move(result));
            continue;
        }

        std::vector<int> indices(view.size());
        std::iota(indices.begin(), indices.end(), 0);

        if (ransac_opts.has_value()) {
            auto ransac_res = ransac<HomographyEstimator>(view, ransac_opts.value());
            if (!ransac_res.success) {
                result.success = false;
                homographies.push_back(std::move(result));
                continue;
            }
            result.hmtx = ransac_res.model;
            result.inliers = std::move(ransac_res.inliers);
            if (std::abs(result.hmtx(2, 2)) > 1e-15) {
                result.hmtx /= result.hmtx(2, 2);
            }
            result.symmetric_rms_px = symmetric_rms_px(result.hmtx, view, result.inliers);
            result.success = true;
        } else {
            auto hopt = HomographyEstimator::fit(view, indices);
            if (!hopt.has_value()) {
                result.success = false;
                homographies.push_back(std::move(result));
                continue;
            }
            result.hmtx = hopt.value();
            if (std::abs(result.hmtx(2, 2)) > 1e-15) {
                result.hmtx /= result.hmtx(2, 2);
            }
            result.inliers = indices;
            result.symmetric_rms_px = symmetric_rms_px(result.hmtx, view, result.inliers);
            result.success = true;
        }

        homographies.push_back(std::move(result));
    }

    return homographies;
}

static auto process_planar_view(const CameraMatrix& kmtx, const HomographyResult& hres)
    -> ViewEstimateData {
    ViewEstimateData ved;
    ved.forward_rms_px = hres.symmetric_rms_px;
    ved.homography = hres;

    auto pose_res = pose_from_homography(kmtx, hres.hmtx);
    if (!pose_res.success) {
        std::cerr << "Warning: Homography decomposition failed: " << pose_res.message << "\n";
    } else {
        ved.c_se3_t = std::move(pose_res.c_se3_t);
    }
    return ved;
}

namespace detail {

auto sanitize_intrinsics(const CameraMatrix& kmtx, const std::optional<CalibrationBounds>& bounds)
    -> std::pair<CameraMatrix, bool> {
    if (!bounds.has_value()) {
        return {kmtx, false};
    }

    const auto& b = bounds.value();
    CameraMatrix adjusted = kmtx;
    bool modified = false;

    const auto enforce_min_focal = [&modified](double value, double min_val) {
        if (!std::isfinite(value) || value < min_val) {
            modified = true;
            return min_val;
        }
        return value;
    };

    const auto midpoint = [](double min_val, double max_val) { return 0.5 * (min_val + max_val); };

    const auto adjust_principal_point = [&modified, &midpoint](double value, double min_val,
                                                               double max_val) {
        if (!std::isfinite(value)) {
            modified = true;
            return midpoint(min_val, max_val);
        }
        if (value < min_val || value > max_val) {
            modified = true;
            return midpoint(min_val, max_val);
        }
        return value;
    };

    adjusted.fx = enforce_min_focal(adjusted.fx, b.fx_min);
    adjusted.fy = enforce_min_focal(adjusted.fy, b.fy_min);
    adjusted.cx = adjust_principal_point(adjusted.cx, b.cx_min, b.cx_max);
    adjusted.cy = adjust_principal_point(adjusted.cy, b.cy_min, b.cy_max);

    const double skew_min = std::min(b.skew_min, b.skew_max);
    const double skew_max = std::max(b.skew_min, b.skew_max);
    if (!std::isfinite(adjusted.skew) || adjusted.skew < skew_min || adjusted.skew > skew_max) {
        modified = true;
        const double zero = 0.0;
        adjusted.skew = std::clamp(zero, skew_min, skew_max);
    }
    return {adjusted, modified};
}

}  // namespace detail

auto estimate_intrinsics(const std::vector<PlanarView>& views,
                         const IntrinsicsEstimateOptions& opts) -> IntrinsicsEstimateResult {
    IntrinsicsEstimateResult result;
    if (views.empty()) {
        return result;
    }

    auto planar_homographies = compute_planar_homographies(views, opts.homography_ransac);

    std::vector<HomographyResult> valid_hs;
    valid_hs.reserve(planar_homographies.size());
    std::copy_if(planar_homographies.begin(), planar_homographies.end(),
                 std::back_inserter(valid_hs),
                 [](const HomographyResult& hr) { return hr.success; });
    auto zhang_kmtx_opt = zhang_intrinsics_from_hs(valid_hs);
    if (!zhang_kmtx_opt.has_value()) {
        std::cout << "Zhang intrinsic estimation failed.\n";
        return result;
    }

    // Set the estimated camera matrix
    auto [sanitized_kmtx, modified] =
        detail::sanitize_intrinsics(zhang_kmtx_opt.value(), opts.bounds);
    result.kmtx = sanitized_kmtx;
    result.success = true;

    if (modified) {
        result.log = "Intrinsics sanitized by bounds.";
    }

    result.views.resize(valid_hs.size());
    std::transform(valid_hs.begin(), valid_hs.end(), result.views.begin(),
                   [&sanitized_kmtx](const HomographyResult& hres) {
                       return process_planar_view(sanitized_kmtx, hres);
                   });
    // fill view indices
    size_t vidx = 0;
    for (size_t i = 0; i < planar_homographies.size(); ++i) {
        if (planar_homographies[i].success) {
            result.views[vidx].view_index = i;
            ++vidx;
        }
    }

    return result;
}

struct LinearSystem final {
    Eigen::MatrixXd amtx;
    Eigen::VectorXd bvec;
};

static auto build_u_system(const std::vector<Observation<double>>& obs, bool use_skew)
    -> LinearSystem {
    const size_t nobs = obs.size();
    const int cols = use_skew ? 3 : 2;

    Eigen::MatrixXd amtx(nobs, cols);
    Eigen::VectorXd bvec(nobs);

    for (size_t i = 0; i < nobs; ++i) {
        const auto& observation = obs[i];
        const int row = static_cast<int>(i);

        amtx(row, 0) = observation.x;  // fx coefficient
        if (use_skew) {
            amtx(row, 1) = observation.y;  // skew coefficient
            amtx(row, 2) = 1.0;            // cx coefficient
        } else {
            amtx(row, 1) = 1.0;  // cx coefficient
        }
        bvec(row) = observation.u;
    }

    return {std::move(amtx), std::move(bvec)};
}

static auto build_v_system(const std::vector<Observation<double>>& obs) -> LinearSystem {
    const size_t nobs = obs.size();

    Eigen::MatrixXd amtx(nobs, 2);
    Eigen::VectorXd bvec(nobs);

    for (size_t i = 0; i < nobs; ++i) {
        const auto& observation = obs[i];
        const int row = static_cast<int>(i);

        amtx(row, 0) = observation.y;  // fy coefficient
        amtx(row, 1) = 1.0;            // cy coefficient
        bvec(row) = observation.v;
    }

    return {std::move(amtx), std::move(bvec)};
}

static auto solve_linear_system(const LinearSystem& system) -> std::optional<Eigen::VectorXd> {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(system.amtx, Eigen::ComputeThinU | Eigen::ComputeThinV);

    if (svd.singularValues().minCoeff() < 1e-12) {
        return std::nullopt;  // Degenerate system
    }

    return svd.solve(system.bvec);
}

static auto apply_bounds_and_fallback(const Eigen::VectorXd& xu, const Eigen::VectorXd& xv,
                                      const std::vector<Observation<double>>& obs,
                                      const CalibrationBounds& bounds, bool use_skew)
    -> CameraMatrix {
    const double fx = xu[0];
    const double fy = xv[0];
    const double cx = use_skew ? xu[2] : xu[1];
    const double cy = xv[1];
    const double skew = use_skew ? xu[1] : 0.0;

    // Check if parameters are within bounds
    const bool out_of_bounds = fx < bounds.fx_min || fx > bounds.fx_max || fy < bounds.fy_min ||
                               fy > bounds.fy_max || cx < bounds.cx_min || cx > bounds.cx_max ||
                               cy < bounds.cy_min || cy > bounds.cy_max ||
                               (use_skew && (skew < bounds.skew_min || skew > bounds.skew_max));

    if (out_of_bounds) {
        std::cerr << "Warning: Linear calibration produced unreasonable intrinsics\n";

        // Compute fallback values
        const double avg_u =
            std::accumulate(obs.begin(), obs.end(), 0.0,
                            [](double sum, const auto& ob) { return sum + ob.u; }) /
            static_cast<double>(obs.size());
        const double avg_v =
            std::accumulate(obs.begin(), obs.end(), 0.0,
                            [](double sum, const auto& ob) { return sum + ob.v; }) /
            static_cast<double>(obs.size());

        const double safe_fx = std::clamp(std::max(500.0, fx), bounds.fx_min, bounds.fx_max);
        const double safe_fy = std::clamp(std::max(500.0, fy), bounds.fy_min, bounds.fy_max);
        const double safe_cx = std::clamp(avg_u / 2.0, bounds.cx_min, bounds.cx_max);
        const double safe_cy = std::clamp(avg_v / 2.0, bounds.cy_min, bounds.cy_max);
        const double safe_skew =
            use_skew ? std::clamp(skew, bounds.skew_min, bounds.skew_max) : 0.0;

        return CameraMatrix{safe_fx, safe_fy, safe_cx, safe_cy, safe_skew};
    }

    return CameraMatrix{fx, fy, cx, cy, skew};
}

static auto correct_observations_for_distortion(const std::vector<Observation<double>>& obs,
                                                const CameraMatrix& kmtx,
                                                const Eigen::VectorXd& distortion)
    -> std::vector<Observation<double>> {
    std::vector<Observation<double>> corrected;
    corrected.reserve(obs.size());

    for (const auto& observation : obs) {
        const Eigen::Vector2d norm(observation.x, observation.y);
        const Eigen::Vector2d distorted = apply_distortion(norm, distortion);
        const Eigen::Vector2d delta = distorted - norm;

        const double u_corr = observation.u - kmtx.fx * delta.x() - kmtx.skew * delta.y();
        const double v_corr = observation.v - kmtx.fy * delta.y();

        corrected.push_back(Observation<double>{observation.x, observation.y, u_corr, v_corr});
    }

    return corrected;
}

static auto compute_camera_matrix_difference(const CameraMatrix& k1, const CameraMatrix& k2)
    -> double {
    return std::abs(k1.fx - k2.fx) + std::abs(k1.fy - k2.fy) + std::abs(k1.cx - k2.cx) +
           std::abs(k1.cy - k2.cy) + std::abs(k1.skew - k2.skew);
}

static auto estimate_distortion_for_camera(const std::vector<Observation<double>>& obs,
                                           const CameraMatrix& kmtx, int num_radial)
    -> std::optional<Eigen::VectorXd> {
    auto dist_opt = fit_distortion(obs, kmtx, num_radial);
    return dist_opt ? std::make_optional(dist_opt->distortion) : std::nullopt;
}

// Compute a linear least-squares estimate of the camera intrinsics
// (fx, fy, cx, cy[, skew]) from normalized correspondences. This ignores lens
// distortion and solves either two or three independent systems depending on
// whether skew is estimated:
//   u = fx * x + skew * y + cx
//   v = fy * y + cy
// If there are insufficient observations or the design matrix is
// degenerate, std::nullopt is returned.
std::optional<CameraMatrix> estimate_intrinsics_linear(const std::vector<Observation<double>>& obs,
                                                       std::optional<CalibrationBounds> bounds_opt,
                                                       bool use_skew) {
    if (obs.size() < 2) {
        return std::nullopt;
    }

    // Build and solve u-equation system: u = fx*x + [skew*y] + cx
    const auto u_system = build_u_system(obs, use_skew);
    const auto xu_opt = solve_linear_system(u_system);
    if (!xu_opt) {
        return std::nullopt;
    }

    // Build and solve v-equation system: v = fy*y + cy
    const auto v_system = build_v_system(obs);
    const auto xv_opt = solve_linear_system(v_system);
    if (!xv_opt) {
        return std::nullopt;
    }

    const CalibrationBounds bounds = bounds_opt.value_or(CalibrationBounds{});
    return apply_bounds_and_fallback(*xu_opt, *xv_opt, obs, bounds, use_skew);
}

// Alternate between fitting distortion coefficients and re-estimating
// the camera matrix.  This provides a better linear initialization for
// subsequent non-linear optimization when moderate distortion is
// present.  If the initial linear estimate fails, std::nullopt is
// returned.
auto estimate_intrinsics_linear_iterative(const std::vector<Observation<double>>& obs,
                                          int num_radial, int max_iterations, bool use_skew)
    -> std::optional<PinholeCamera<BrownConradyd>> {
    // Get initial linear estimate without distortion
    auto kmtx_opt = estimate_intrinsics_linear(obs, std::nullopt, use_skew);
    if (!kmtx_opt) {
        return std::nullopt;
    }

    CameraMatrix kmtx = *kmtx_opt;
    constexpr double convergence_threshold = 1e-6;

    // Iteratively refine camera matrix by accounting for distortion
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Estimate distortion for current camera matrix
        const auto distortion_opt = estimate_distortion_for_camera(obs, kmtx, num_radial);
        if (!distortion_opt) {
            break;  // Failed to estimate distortion
        }

        // Correct observations by removing estimated distortion
        const auto corrected_obs = correct_observations_for_distortion(obs, kmtx, *distortion_opt);

        // Re-estimate camera matrix using corrected observations
        const auto kmtx_new_opt = estimate_intrinsics_linear(corrected_obs, std::nullopt, use_skew);
        if (!kmtx_new_opt) {
            break;  // Failed to re-estimate camera matrix
        }

        // Check for convergence
        const double change = compute_camera_matrix_difference(kmtx, *kmtx_new_opt);
        kmtx = *kmtx_new_opt;

        if (change < convergence_threshold) {
            break;  // Converged
        }
    }

    // Final distortion estimation with refined camera matrix
    const auto final_distortion_opt = fit_distortion_full(obs, kmtx, num_radial);
    if (!final_distortion_opt) {
        return std::nullopt;
    }

    PinholeCamera<BrownConradyd> camera;
    camera.kmtx = kmtx;
    camera.distortion.coeffs = final_distortion_opt->distortion;

    return camera;
}

}  // namespace calib
