#include "calib/intrinsics.h"

// std
#include <numeric>
#include <optional>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "ceresutils.h"
#include "observationutils.h"
#include "residuals/intrinsicsemidltresidual.h"

namespace calib {

static size_t count_total_observations(const std::vector<PlanarView>& views) {
    size_t total_obs = 0;
    for (const auto& view : views) {
        total_obs += view.size();
    }
    return total_obs;
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
            Eigen::Isometry3d pose = estimate_planar_pose_dlt(views[i], initial_guess);
            populate_quat_tran(pose, blocks.c_quat_t[i], blocks.c_tra_t[i]);
        }

        return blocks;
    }

    std::vector<ParamBlock> get_param_blocks() const override {
        std::vector<ParamBlock> blocks;
        blocks.emplace_back(ParamBlock{intrinsics.data(), intrinsics.size(), 5});
        for (const auto& q : c_quat_t) {
            blocks.emplace_back(ParamBlock{q.data(), q.size(), 3});
        }
        for (const auto& t : c_tra_t) {
            blocks.emplace_back(ParamBlock{t.data(), t.size(), 3});
        }
        return blocks;
    }

    void populate_result(IntrinsicsOptimizationResult<Camera<BrownConradyd>>& result) const {
        result.camera.K.fx = intrinsics[0];
        result.camera.K.fy = intrinsics[1];
        result.camera.K.cx = intrinsics[2];
        result.camera.K.cy = intrinsics[3];
        result.camera.K.skew = intrinsics[4];

        result.c_se3_t.resize(c_quat_t.size());
        for (size_t i = 0; i < c_quat_t.size(); ++i) {
            result.c_se3_t[i] = restore_pose(c_quat_t[i], c_tra_t[i]);
        }
    }
};

static auto solve_full(const std::vector<PlanarView>& views, int num_radial,
                       const IntrinsicBlocks& blocks)
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
    return fit_distortion_full(obs, kmtx, num_radial);
}

// Set up the Ceres optimization problem
static ceres::Problem build_problem(const std::vector<PlanarView>& obs_views,
                                    IntrinsicBlocks& blocks, const IntrinsicsOptions& opts) {
    ceres::Problem problem;
    auto* cost = CalibVPResidual::create(obs_views, opts.num_radial);

    // Add parameter blocks to the problem
    std::vector<double*> param_blocks;
    param_blocks.push_back(blocks.intrinsics.data());
    for (size_t i = 0; i < blocks.c_quat_t.size(); ++i) {
        param_blocks.push_back(blocks.c_quat_t[i].data());
        param_blocks.push_back(blocks.c_tra_t[i].data());
    }
    auto loss = opts.huber_delta > 0 ? new ceres::HuberLoss(opts.huber_delta) : nullptr;
    problem.AddResidualBlock(cost, loss, param_blocks);

    for (size_t i = 0; i < blocks.c_quat_t.size(); ++i) {
        problem.SetManifold(blocks.c_quat_t[i].data(), new ceres::QuaternionManifold());
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

static void compute_per_view_errors(const std::vector<PlanarView>& obs_views,
                                    const Eigen::VectorXd& residuals,
                                    IntrinsicsOptimizationResult<Camera<BrownConradyd>>& result) {
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

IntrinsicsOptimizationResult<Camera<BrownConradyd>> optimize_intrinsics_semidlt(
    const std::vector<PlanarView>& views, const CameraMatrix& initial_guess,
    const IntrinsicsOptions& opts) {
    IntrinsicsOptimizationResult<Camera<BrownConradyd>> result;

    // Prepare observations per view
    const size_t total_obs = count_total_observations(views);
    const size_t num_views = views.size();
    if (num_views < 4) {
        std::cerr << "Insufficient views for calibration (at least 4 required)." << std::endl;
        return result;
    }

    auto blocks = IntrinsicBlocks::create(views, initial_guess);
    // Set up and solve the optimization problem
    ceres::Problem problem = build_problem(views, blocks, opts);

    solve_problem(problem, opts, &result);

    auto dr_opt = solve_full(views, opts.num_radial, blocks);
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
