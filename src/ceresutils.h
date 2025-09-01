/** @brief Ceres solver utilities */

#pragma once

// std
#include <cmath>
#include <optional>
#include <thread>

// eigen
#include <Eigen/Core>

// ceres
#include "calib/optimize.h"
#include "ceres/covariance.h"
#include "ceres/solver.h"

namespace calib {

static const std::map<OptimizerType, ceres::LinearSolverType> optim_to_ceres = {
    {OptimizerType::DEFAULT, ceres::SPARSE_NORMAL_CHOLESKY},
    {OptimizerType::SPARSE_SCHUR, ceres::SPARSE_SCHUR},
    {OptimizerType::DENSE_SCHUR, ceres::DENSE_SCHUR},
    {OptimizerType::DENSE_QR, ceres::DENSE_QR}};

inline void solve_problem(ceres::Problem& problem, const OptimOptions& opts, OptimResult* result) {
    ceres::Solver::Options copts;
    copts.linear_solver_type = optim_to_ceres.at(opts.optimizer);
    copts.num_threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    copts.minimizer_progress_to_stdout = opts.verbose;
    copts.function_tolerance = opts.epsilon;
    copts.gradient_tolerance = opts.epsilon;
    copts.parameter_tolerance = opts.epsilon;
    copts.max_num_iterations = opts.max_iterations;

    ceres::Solver::Summary summary;
    ceres::Solve(copts, &problem, &summary);

    result->final_cost = summary.final_cost;
    result->report = summary.BriefReport();
    result->success = summary.termination_type == ceres::CONVERGENCE;
}

struct ParamBlock final {
    const double* data;
    size_t size;
    size_t dof;

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    ParamBlock(const double* data_ptr, size_t block_size, size_t block_dof)
        : data(data_ptr), size(block_size), dof(block_dof) {}
};

struct ProblemParamBlocks {
    [[nodiscard]]
    virtual auto get_param_blocks() const -> std::vector<ParamBlock> = 0;

    [[nodiscard]]
    auto total_params() const -> size_t {
        size_t total_count = 0;
        for (const auto& param_block : get_param_blocks()) {
            total_count += param_block.size;
        }
        return total_count;
    }
};

// Compute and populate the covariance matrix
inline auto compute_covariance(
    const ProblemParamBlocks& problem_param_blocks, ceres::Problem& problem,
    double sum_squared_residuals = 0, size_t total_residuals = 0) -> std::optional<Eigen::MatrixXd> { // NOLINT(bugprone-easily-swappable-parameters)
    auto param_blocks = problem_param_blocks.get_param_blocks();
    const size_t total_params = problem_param_blocks.total_params();

    ceres::Covariance::Options cov_options;
    ceres::Covariance covariance(cov_options);
    std::vector<std::pair<const double*, const double*>> cov_blocks;

    for (size_t block_i = 0; block_i < param_blocks.size(); ++block_i) {
        for (size_t block_j = 0; block_j <= block_i; ++block_j) {
            cov_blocks.emplace_back(param_blocks[block_i].data, param_blocks[block_j].data);
        }
    }

    if (!covariance.Compute(cov_blocks, &problem)) {
        return std::nullopt;
    }

    Eigen::MatrixXd cov_matrix = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(total_params),
                                                       static_cast<Eigen::Index>(total_params));
    size_t row_offset = 0;
    for (size_t block_i = 0; block_i < param_blocks.size(); ++block_i) {
        size_t block_i_size = param_blocks[block_i].size;
        size_t col_offset = 0;
        for (size_t block_j = 0; block_j <= block_i; ++block_j) {
            size_t block_j_size = param_blocks[block_j].size;
            std::vector<double> block_cov(block_i_size * block_j_size);
            covariance.GetCovarianceBlock(param_blocks[block_i].data, param_blocks[block_j].data,
                                          block_cov.data());
            for (size_t row_idx = 0; row_idx < block_i_size; ++row_idx) {
                for (size_t col_idx = 0; col_idx < block_j_size; ++col_idx) {
                    double value = block_cov[row_idx * block_j_size + col_idx];
                    cov_matrix(static_cast<Eigen::Index>(row_offset + row_idx),
                               static_cast<Eigen::Index>(col_offset + col_idx)) = value;
                    if (block_j < block_i) {
                        cov_matrix(static_cast<Eigen::Index>(col_offset + col_idx),
                                   static_cast<Eigen::Index>(row_offset + row_idx)) = value;
                    }
                }
            }
            col_offset += block_j_size;
        }
        row_offset += block_i_size;
    }

    if (total_residuals > 0) {
        // Scale covariance by variance factor
        int degrees_of_freedom =
            std::max(1, static_cast<int>(total_residuals) - static_cast<int>(total_params));
        double variance_factor = sum_squared_residuals / degrees_of_freedom;
        cov_matrix *= variance_factor;
    }

    return cov_matrix;
}

}  // namespace calib
