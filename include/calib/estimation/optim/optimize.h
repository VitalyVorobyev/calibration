#pragma once

// std
#include <cstdint>
#include <string>

// Eigen
#include <Eigen/Core>

namespace calib {

enum class OptimizerType : uint8_t { DEFAULT, SPARSE_SCHUR, DENSE_SCHUR, DENSE_QR };

// TODO: use nlohmann macros for enum serialization
inline auto optimizer_type_to_string(OptimizerType type) -> std::string {
    switch (type) {
        case OptimizerType::DEFAULT:
            return "default";
        case OptimizerType::SPARSE_SCHUR:
            return "sparse_schur";
        case OptimizerType::DENSE_SCHUR:
            return "dense_schur";
        case OptimizerType::DENSE_QR:
            return "dense_qr";
    }
    return "default";
}

inline auto optimizer_type_from_string(std::string value) -> OptimizerType {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "default") return OptimizerType::DEFAULT;
    if (value == "sparse_schur") return OptimizerType::SPARSE_SCHUR;
    if (value == "dense_schur") return OptimizerType::DENSE_SCHUR;
    if (value == "dense_qr") return OptimizerType::DENSE_QR;
    throw std::runtime_error("Unknown optimizer type: " + value);
}

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
