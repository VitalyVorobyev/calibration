/**
 * @file intrinsics.h
 * @brief Camera intrinsic parameter estimation and optimization
 * @ingroup camera_calibration
 *
 * This file provides comprehensive intrinsic camera calibration functionality including:
 * - Linear least squares estimation (DLT)
 * - Semi-linear iterative refinement
 * - Bundle adjustment with distortion correction
 * - Uncertainty quantification through covariance estimation
 * - Support for various camera models and constraints
 */

#pragma once

// std
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/camera.h"
#include "calib/cameramodel.h"
#include "calib/optimize.h"
#include "calib/planarpose.h"

namespace calib {

/**
 * @brief Estimate camera intrinsics using linear least squares
 * @ingroup camera_calibration
 *
 * Estimates camera intrinsic parameters (fx, fy, cx, cy, optionally skew)
 * by solving a linear least-squares system that ignores lens distortion.
 * The input observations contain normalized coordinates (x,y) for undistorted
 * points and observed pixel coordinates (u,v).
 *
 * @param observations Vector of distortion observations
 * @param bounds Optional calibration parameter bounds
 * @param use_skew Whether to estimate skew parameter
 * @return Optional camera matrix (nullopt if insufficient data or degenerate system)
 *
 * @note Requires at least 6 observations for overdetermined system
 * @note Ignores lens distortion - suitable for initial estimation only
 */
auto estimate_intrinsics_linear(const std::vector<Observation<double>>& observations,
                                std::optional<CalibrationBounds> bounds = std::nullopt,
                                bool use_skew = false) -> std::optional<CameraMatrix>;

/**
 * @brief Improved linear initialization with distortion estimation
 * @ingroup camera_calibration
 *
 * Iterative algorithm that alternates between estimating distortion coefficients
 * and re-solving for camera intrinsics to provide better initial estimates.
 * Returns std::nullopt if the initial linear estimation fails. The number of
 * radial distortion coefficients and the number of refinement iterations can be specified.
 */
constexpr int k_default_max_iterations = 5;
auto estimate_intrinsics_linear_iterative(
    const std::vector<Observation<double>>& observations, int num_radial,
    int max_iterations = k_default_max_iterations,
    bool use_skew = false) -> std::optional<Camera<BrownConradyd>>;

struct IntrinsicsOptions final : public OptimOptions {
    int num_radial = 2;          ///< Number of radial distortion coefficients
    bool optimize_skew = false;  ///< Estimate skew parameter
    std::optional<CalibrationBounds> bounds = std::nullopt;  ///< Parameter bounds
};

template <camera_model CameraT>
struct IntrinsicsOptimizationResult final : public OptimResult {
    CameraT camera;                          ///< Estimated camera parameters
    std::vector<Eigen::Isometry3d> c_se3_t;  ///< Estimated pose of each view
    std::vector<double> view_errors;         ///< Per-view reprojection errors
};

auto optimize_intrinsics_semidlt(
    const std::vector<PlanarView>& views, const CameraMatrix& initial_guess,
    const IntrinsicsOptions& opts = {}) -> IntrinsicsOptimizationResult<Camera<BrownConradyd>>;

template <camera_model CameraT>
auto optimize_intrinsics(const std::vector<PlanarView>& views, const CameraT& init_camera,
                         std::vector<Eigen::Isometry3d> init_c_se3_t,
                         const IntrinsicsOptions& opts = {})
    -> IntrinsicsOptimizationResult<CameraT>;

}  // namespace calib
