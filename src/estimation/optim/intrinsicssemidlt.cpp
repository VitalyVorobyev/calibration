#include "calib/estimation/optim/intrinsics.h"

// std
#include <algorithm>
#include <numeric>
#include <optional>
#include <span>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "detail/ceresutils.h"
#include "detail/observationutils.h"
#include "residuals/intrinsicsemidltresidual.h"

namespace calib {

static size_t count_total_observations(const std::vector<PlanarView>& views) {
    return std::accumulate(views.begin(), views.end(), size_t{0},
                           [](size_t total, const auto& view) { return total + view.size(); });
}

struct IntrinsicBlocks final : public ProblemParamBlocks {
    std::array<double, 5> intrinsics;
    std::vector<std::array<double, 4>> c_quat_t;
    std::vector<std::array<double, 3>> c_tra_t;

    static IntrinsicBlocks create(const std::vector<PlanarView>& views,
                                  const CameraMatrix& initial_guess) {
        IntrinsicBlocks blocks;
        blocks.c_quat_t.resize(views.size());
        blocks.c_tra_t.resize(views.size());
        blocks.intrinsics = {initial_guess.fx, initial_guess.fy, initial_guess.cx, initial_guess.cy,
                             initial_guess.skew};

        for (size_t i = 0; i < views.size(); ++i) {
            Eigen::Isometry3d pose = estimate_planar_pose(views[i], initial_guess);
            populate_quat_tran(pose, blocks.c_quat_t[i], blocks.c_tra_t[i]);
        }

        return blocks;
    }

    [[nodiscard]] std::vector<ParamBlock> get_param_blocks() const override {
        std::vector<ParamBlock> blocks;
        blocks.emplace_back(intrinsics.data(), intrinsics.size(), 5);

        // Reserve space for efficiency
        blocks.reserve(1 + c_quat_t.size() + c_tra_t.size());

        // Add quaternion blocks using std::transform
        std::transform(c_quat_t.begin(), c_quat_t.end(), std::back_inserter(blocks),
                       [](const auto& q) { return ParamBlock{q.data(), q.size(), 3}; });

        // Add translation blocks using std::transform
        std::transform(c_tra_t.begin(), c_tra_t.end(), std::back_inserter(blocks),
                       [](const auto& t) { return ParamBlock{t.data(), t.size(), 3}; });

        return blocks;
    }

    void populate_result(IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>>& result) const {
        result.camera.kmtx.fx = intrinsics[0];
        result.camera.kmtx.fy = intrinsics[1];
        result.camera.kmtx.cx = intrinsics[2];
        result.camera.kmtx.cy = intrinsics[3];
        result.camera.kmtx.skew = intrinsics[4];

        result.c_se3_t.resize(c_quat_t.size());
        for (size_t i = 0; i < c_quat_t.size(); ++i) {
            result.c_se3_t[i] = restore_pose(c_quat_t[i], c_tra_t[i]);
        }
    }
};

static auto solve_full(const std::vector<PlanarView>& views, const IntrinsicBlocks& blocks,
                       const IntrinsicsOptimOptions& opts)
    -> std::optional<DistortionWithResiduals<double>> {
    std::vector<Observation<double>> obs;
    for (size_t i = 0; i < views.size(); ++i) {
        auto c_se3_t = restore_pose(blocks.c_quat_t[i], blocks.c_tra_t[i]);
        std::vector<Observation<double>> new_obs(views[i].size());
        planar_observables_to_observables(views[i], new_obs, c_se3_t);
        obs.insert(obs.end(), new_obs.begin(), new_obs.end());
    }
    CameraMatrix kmtx{blocks.intrinsics[0], blocks.intrinsics[1], blocks.intrinsics[2],
                      blocks.intrinsics[3], blocks.intrinsics[4]};
    return fit_distortion_full(obs, kmtx, opts.num_radial,
                               std::span<const int>(opts.fixed_distortion_indices),
                               std::span<const double>(opts.fixed_distortion_values));
}

// Set up the Ceres optimization problem
static ceres::Problem build_problem(const std::vector<PlanarView>& obs_views,
                                    IntrinsicBlocks& blocks, const IntrinsicsOptimOptions& opts) {
    ceres::Problem problem;
    auto* cost = CalibVPResidual::create(obs_views, opts.num_radial);

    // Add parameter blocks to the problem
    std::vector<double*> param_blocks;
    param_blocks.push_back(blocks.intrinsics.data());
    for (size_t i = 0; i < blocks.c_quat_t.size(); ++i) {
        param_blocks.push_back(blocks.c_quat_t[i].data());
        param_blocks.push_back(blocks.c_tra_t[i].data());
    }
    auto* loss = opts.huber_delta > 0 ? new ceres::HuberLoss(opts.huber_delta) : nullptr;
    problem.AddResidualBlock(cost, loss, param_blocks);

    for (auto& i : blocks.c_quat_t) {
        problem.SetManifold(i.data(), new ceres::QuaternionManifold());
    }

    if (opts.bounds.has_value()) {
        CalibrationBounds bounds = opts.bounds.value_or(CalibrationBounds{});
        using Traits = CameraTraits<Camera<BrownConradyd>>;
        problem.SetParameterLowerBound(blocks.intrinsics.data(), Traits::idx_fx, bounds.fx_min);
        problem.SetParameterLowerBound(blocks.intrinsics.data(), Traits::idx_fy, bounds.fy_min);
        problem.SetParameterLowerBound(blocks.intrinsics.data(), 2, bounds.cx_min);
        problem.SetParameterLowerBound(blocks.intrinsics.data(), 3, bounds.cy_min);
        problem.SetParameterLowerBound(blocks.intrinsics.data(), Traits::idx_skew, bounds.skew_min);
        problem.SetParameterUpperBound(blocks.intrinsics.data(), Traits::idx_skew, bounds.skew_max);
        problem.SetParameterUpperBound(blocks.intrinsics.data(), Traits::idx_fx, bounds.fx_max);
        problem.SetParameterUpperBound(blocks.intrinsics.data(), Traits::idx_fy, bounds.fy_max);
        problem.SetParameterUpperBound(blocks.intrinsics.data(), 2, bounds.cx_max);
        problem.SetParameterUpperBound(blocks.intrinsics.data(), 3, bounds.cy_max);
    }
    if (!opts.optimize_skew) {
        problem.SetManifold(
            blocks.intrinsics.data(),
            new ceres::SubsetManifold(5, {CameraTraits<Camera<BrownConradyd>>::idx_skew}));
    }

    return problem;
}

static void compute_per_view_errors(
    const std::vector<PlanarView>& obs_views, const Eigen::VectorXd& residuals,
    IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>>& result) {
    const size_t num_views = obs_views.size();
    result.view_errors.resize(num_views);
    int residual_idx = 0;

    for (size_t i = 0; i < num_views; ++i) {
        int view_points = static_cast<int>(obs_views[i].size());
        double view_error_sum = 0.0;
        for (int j = 0; j < view_points * 2; ++j) {
            double r = residuals[residual_idx++];
            view_error_sum += r * r;
        }
        result.view_errors[i] = std::sqrt(view_error_sum / (view_points * 2));
    }
}

IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>> optimize_intrinsics_semidlt(
    const std::vector<PlanarView>& views, const CameraMatrix& initial_guess,
    const IntrinsicsOptimOptions& opts) {
    IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>> result;

    // Prepare observations per view
    const size_t total_obs = count_total_observations(views);
    const size_t num_views = views.size();
    if (num_views < 4) {
        std::cerr << "Insufficient views for calibration (at least 4 required)." << '\n';
        return result;
    }

    auto blocks = IntrinsicBlocks::create(views, initial_guess);
    // Set up and solve the optimization problem
    ceres::Problem problem = build_problem(views, blocks, opts);

    solve_problem(problem, opts, &result);

    auto dr_opt = solve_full(views, blocks, opts);
    if (!dr_opt.has_value()) {
        throw std::runtime_error("Failed to compute distortion parameters");
    }
    result.camera.distortion.coeffs = dr_opt->distortion;

    // Process results
    compute_per_view_errors(views, dr_opt->residuals, result);
    blocks.populate_result(result);

    double sum_squared_residuals = dr_opt->residuals.squaredNorm();
    size_t total_residuals = total_obs * 2;
    result.covariance = compute_covariance(blocks, problem, sum_squared_residuals, total_residuals)
                            .value_or(Eigen::MatrixXd{});

    return result;
}

}  // namespace calib
