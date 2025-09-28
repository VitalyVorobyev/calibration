#pragma once

// std
#include <cstdint>
#include <string>

// Eigen
#include <Eigen/Core>

// nlohmann
#include <nlohmann/json.hpp>

namespace calib {

enum class OptimizerType : uint8_t { DEFAULT, SPARSE_SCHUR, DENSE_SCHUR, DENSE_QR };

NLOHMANN_JSON_SERIALIZE_ENUM(OptimizerType,
                             {{OptimizerType::DEFAULT, "default"},
                              {OptimizerType::SPARSE_SCHUR, "sparse_schur"},
                              {OptimizerType::DENSE_SCHUR, "dense_schur"},
                              {OptimizerType::DENSE_QR, "dense_qr"}})

struct OptimOptions {
    OptimizerType optimizer = OptimizerType::DEFAULT;
    double huber_delta = 1.0;
    static constexpr double k_default_epsilon = 1e-9;
    double epsilon = k_default_epsilon;
    static constexpr int k_default_max_iterations = 1000;
    int max_iterations = k_default_max_iterations;
    bool compute_covariance = true;
    bool verbose = false;
};

struct OptimResult {
    bool success = false;
    Eigen::MatrixXd covariance;
    std::string report = "Empty";
    double final_cost = 0.0;
};

}  // namespace calib
