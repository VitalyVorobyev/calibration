#pragma once

// std
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>

// Eigen
#include <Eigen/Core>

#include "calib/io/serialization.h"  // to_json/from_json

namespace calib {

enum class OptimizerType { DEFAULT, SPARSE_SCHUR, DENSE_SCHUR, DENSE_QR };

NLOHMANN_JSON_SERIALIZE_ENUM(OptimizerType, {{OptimizerType::DEFAULT, "default"},
                                             {OptimizerType::SPARSE_SCHUR, "sparse_schur"},
                                             {OptimizerType::DENSE_SCHUR, "dense_schur"},
                                             {OptimizerType::DENSE_QR, "dense_qr"}})

struct OptimOptions final {
    static constexpr double k_default_epsilon = 1e-9;
    static constexpr int k_default_max_iterations = 1000;
    OptimizerType optimizer = OptimizerType::DEFAULT;
    double huber_delta = 1.0;
    double epsilon = k_default_epsilon;
    int max_iterations = k_default_max_iterations;
    bool compute_covariance = true;
    bool verbose = false;
};

struct OptimResult final {
    bool success = false;
    Eigen::MatrixXd covariance;
    std::string report = "Empty";
    double final_cost = 0.0;
};

static_assert(serializable_aggregate<OptimOptions>);
static_assert(serializable_aggregate<OptimResult>);

}  // namespace calib
