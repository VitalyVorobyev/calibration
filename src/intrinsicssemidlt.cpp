#include "calib/intrinsics.h"

// std
#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_set>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "ceresutils.h"
#include "observationutils.h"
#include "residuals/intrinsicsemidltresidual.h"

namespace calib {

static size_t count_total_observations(const std::vector<PlanarView>& views) {
    return std::accumulate(views.begin(), views.end(), size_t{0},
                           [](size_t total, const auto& view) { return total + view.size(); });
}

static auto build_distortion_system(const std::vector<Observation<double>>& observations,
                                    const CameraMatrix& intrinsics, int num_radial)
    -> std::pair<Eigen::MatrixXd, Eigen::VectorXd> {
    const int num_coeffs = num_radial + 2;
    const int num_obs = static_cast<int>(observations.size());
    const int num_rows = num_obs * 2;

    Eigen::MatrixXd design_matrix = Eigen::MatrixXd::Zero(num_rows, num_coeffs);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(num_rows);

    const int idx_p1 = num_radial;
    const int idx_p2 = num_radial + 1;

    for (int obs_idx = 0; obs_idx < num_obs; ++obs_idx) {
        const double x_coord = observations[obs_idx].x;
        const double y_coord = observations[obs_idx].y;
        const double r2_val = x_coord * x_coord + y_coord * y_coord;

        const double undistorted_u =
            intrinsics.fx * x_coord + intrinsics.skew * y_coord + intrinsics.cx;
        const double undistorted_v = intrinsics.fy * y_coord + intrinsics.cy;

        const double residual_u = observations[obs_idx].u - undistorted_u;
        const double residual_v = observations[obs_idx].v - undistorted_v;

        const int row_u = 2 * obs_idx;
        const int row_v = row_u + 1;

        double rpow = r2_val;
        for (int j = 0; j < num_radial; ++j) {
            design_matrix(row_u, j) =
                intrinsics.fx * x_coord * rpow + intrinsics.skew * y_coord * rpow;
            design_matrix(row_v, j) = intrinsics.fy * y_coord * rpow;
            rpow *= r2_val;
        }

        design_matrix(row_u, idx_p1) = intrinsics.fx * (2.0 * x_coord * y_coord) +
                                       intrinsics.skew * (r2_val + 2.0 * y_coord * y_coord);
        design_matrix(row_u, idx_p2) = intrinsics.fx * (r2_val + 2.0 * x_coord * x_coord) +
                                       intrinsics.skew * (2.0 * x_coord * y_coord);
        design_matrix(row_v, idx_p1) = intrinsics.fy * (r2_val + 2.0 * y_coord * y_coord);
        design_matrix(row_v, idx_p2) = intrinsics.fy * (2.0 * x_coord * y_coord);

        rhs(row_u) = residual_u;
        rhs(row_v) = residual_v;
    }

    return {std::move(design_matrix), std::move(rhs)};
}

static auto solve_distortion(const std::vector<Observation<double>>& observations,
                             const CameraMatrix& intrinsics, const IntrinsicsOptions& opts)
    -> std::optional<DistortionWithResiduals<double>> {
    constexpr int k_min_observations = 8;
    if (static_cast<int>(observations.size()) < k_min_observations) {
        return std::nullopt;
    }

    const int num_coeffs = opts.num_radial + 2;
    auto [design_matrix, rhs] = build_distortion_system(observations, intrinsics, opts.num_radial);

    if (opts.fixed_distortion_indices.empty()) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(design_matrix,
                                              Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd alpha = svd.solve(rhs);
        Eigen::VectorXd residuals = design_matrix * alpha - rhs;
        return DistortionWithResiduals<double>{std::move(alpha), std::move(residuals)};
    }

    std::vector<std::pair<int, double>> fixed_pairs;
    fixed_pairs.reserve(opts.fixed_distortion_indices.size());
    for (size_t i = 0; i < opts.fixed_distortion_indices.size(); ++i) {
        int idx = opts.fixed_distortion_indices[i];
        double value = 0.0;
        if (i < opts.fixed_distortion_values.size()) {
            value = opts.fixed_distortion_values[i];
        }
        fixed_pairs.emplace_back(idx, value);
    }
    std::sort(fixed_pairs.begin(), fixed_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    fixed_pairs.erase(std::unique(fixed_pairs.begin(), fixed_pairs.end(),
                                  [](const auto& a, const auto& b) { return a.first == b.first; }),
                      fixed_pairs.end());

    std::vector<int> fixed_indices_only;
    fixed_indices_only.reserve(fixed_pairs.size());
    for (const auto& [idx, _] : fixed_pairs) {
        fixed_indices_only.push_back(idx);
    }

    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(num_coeffs);
    for (const auto& [idx, value] : fixed_pairs) {
        if (idx < 0 || idx >= num_coeffs) {
            throw std::invalid_argument("Fixed distortion index out of range");
        }
        coeffs(idx) = value;
    }

    Eigen::VectorXd rhs_adjusted = rhs;
    for (const auto& [idx, value] : fixed_pairs) {
        rhs_adjusted -= design_matrix.col(idx) * value;
    }

    std::vector<int> free_indices;
    free_indices.reserve(num_coeffs - static_cast<int>(fixed_pairs.size()));
    for (int idx = 0; idx < num_coeffs; ++idx) {
        if (!std::binary_search(fixed_indices_only.begin(), fixed_indices_only.end(), idx)) {
            free_indices.push_back(idx);
        }
    }

    Eigen::VectorXd residuals;
    if (free_indices.empty()) {
        residuals = design_matrix * coeffs - rhs;
        return DistortionWithResiduals<double>{std::move(coeffs), std::move(residuals)};
    }

    Eigen::MatrixXd free_design(design_matrix.rows(), static_cast<int>(free_indices.size()));
    for (int col = 0; col < static_cast<int>(free_indices.size()); ++col) {
        free_design.col(col) = design_matrix.col(free_indices[col]);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(free_design, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd free_alpha = svd.solve(rhs_adjusted);

    for (int col = 0; col < static_cast<int>(free_indices.size()); ++col) {
        coeffs(free_indices[col]) = free_alpha(col);
    }

    residuals = design_matrix * coeffs - rhs;
    return DistortionWithResiduals<double>{std::move(coeffs), std::move(residuals)};
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
                       const IntrinsicsOptions& opts)
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
    return solve_distortion(obs, kmtx, opts);
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
    const IntrinsicsOptions& opts) {
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
