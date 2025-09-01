/** @brief Ceres solver utilities */

#pragma once

// std
#include <cmath>
#include <thread>
#include <optional>

// eigen
#include <Eigen/Core>

// ceres
#include "ceres/solver.h"
#include "ceres/covariance.h"

#include "calib/optimize.h"

namespace calib {

static const std::map<OptimizerType, ceres::LinearSolverType> optim_to_ceres = {
    { OptimizerType::DEFAULT, ceres::SPARSE_NORMAL_CHOLESKY },
    { OptimizerType::SPARSE_SCHUR, ceres::SPARSE_SCHUR },
    { OptimizerType::DENSE_SCHUR, ceres::DENSE_SCHUR },
    { OptimizerType::DENSE_QR, ceres::DENSE_QR }
};

inline void solve_problem(ceres::Problem& p, const OptimOptions& opts, OptimResult* result) {
    ceres::Solver::Options copts;
    copts.linear_solver_type = optim_to_ceres.at(opts.optimizer);
    copts.num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    copts.minimizer_progress_to_stdout = opts.verbose;
    copts.function_tolerance = opts.epsilon;
    copts.gradient_tolerance = opts.epsilon;
    copts.parameter_tolerance = opts.epsilon;
    copts.max_num_iterations = opts.max_iterations;

    ceres::Solver::Summary summary;
    ceres::Solve(copts, &p, &summary);

    result->final_cost = summary.final_cost;
    result->report = summary.BriefReport();
    result->success = summary.termination_type == ceres::CONVERGENCE;
}

struct ParamBlock final {
    const double *data;
    size_t size;
    size_t dof;
    
    ParamBlock(const double* data_, size_t size_, size_t dof_) 
        : data(data_), size(size_), dof(dof_) {}
};

struct ProblemParamBlocks {
    virtual std::vector<ParamBlock> get_param_blocks() const = 0;

    size_t total_params() const {
        size_t total = 0;
        for (const auto& block : get_param_blocks()) {
            total += block.size;
        }
        return total;
    }
};

// Compute and populate the covariance matrix
inline std::optional<Eigen::MatrixXd> compute_covariance(
    const ProblemParamBlocks& problem_param_blocks,
    ceres::Problem& problem,
    size_t total_residuals = 0,
    double sum_squared_residuals = 0
) {
    auto param_blocks = problem_param_blocks.get_param_blocks();
    const size_t total_params = problem_param_blocks.total_params();

    ceres::Covariance::Options cov_options;
    ceres::Covariance covariance(cov_options);
    std::vector<std::pair<const double*, const double*>> cov_blocks;

    for (size_t i = 0; i < param_blocks.size(); ++i) {
        for (size_t j = 0; j <= i; ++j) {
            cov_blocks.emplace_back(param_blocks[i].data, param_blocks[j].data);
        }
    }

    if (!covariance.Compute(cov_blocks, &problem)) {
        return std::nullopt;
    }

    Eigen::MatrixXd cov_matrix = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(total_params), static_cast<Eigen::Index>(total_params));
    size_t row_offset = 0;

    for (size_t i = 0; i < param_blocks.size(); ++i) {
        size_t block_i_size = param_blocks[i].size;
        size_t col_offset = 0;

        for (size_t j = 0; j <= i; ++j) {
            size_t block_j_size = param_blocks[j].size;
            std::vector<double> block_cov(block_i_size * block_j_size);

            covariance.GetCovarianceBlock(
                param_blocks[i].data,
                param_blocks[j].data,
                block_cov.data());

            for (size_t r = 0; r < block_i_size; ++r) {
                for (size_t c = 0; c < block_j_size; ++c) {
                    double value = block_cov[r * block_j_size + c];
                    cov_matrix(static_cast<Eigen::Index>(row_offset + r), static_cast<Eigen::Index>(col_offset + c)) = value;
                    if (j < i) {
                        cov_matrix(static_cast<Eigen::Index>(col_offset + c), static_cast<Eigen::Index>(row_offset + r)) = value;
                    }
                }
            }
            col_offset += block_j_size;
        }
        row_offset += block_i_size;
    }

    if (total_residuals > 0) {
        // Scale covariance by variance factor
        int degrees_of_freedom = std::max(1, static_cast<int>(total_residuals) - static_cast<int>(total_params));
        double variance_factor = sum_squared_residuals / degrees_of_freedom;
        cov_matrix *= variance_factor;
    }

    return cov_matrix;
}

}  // namespace calib
