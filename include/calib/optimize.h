/** @brief Common code for optimization routines */

#pragma once

namespace calib {

enum class OptimizerType {
    DEFAULT,       // SPARSE_NORMAL_CHOLESKY
    SPARSE_SCHUR,  // for large problems
    DENSE_SCHUR,   // for small multiple camera problems
    DENSE_QR       // for small single camera problems
};

struct OptimOptions {
    OptimizerType optimizer = OptimizerType::DEFAULT;
    double huber_delta = 1.0;        ///< Huber loss delta. L2 loss is used if huber_delta < 0
    double epsilon = 1e-9;           ///< Solver convergence tolerance
    int max_iterations = 1000;       ///< Maximum number of iterations
    bool compute_covariance = true;  ///< Compute covariance matrix
    bool verbose = false;            ///< Verbose solver output
};

struct OptimResult {
    bool success = false;
    Eigen::MatrixXd covariance;
    std::string report = "Empty";  ///< Solver brief report
    double final_cost = 0.0;
};

}  // namespace calib
