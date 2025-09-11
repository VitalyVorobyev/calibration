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
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/cameramodel.h"
#include "calib/optimize.h"
#include "calib/pinhole.h"
#include "calib/planarpose.h"

namespace calib {

/**
 * @brief Options for linear intrinsic estimation from planar views
 * @ingroup camera_calibration
 *
 * Controls bounds and whether skew is estimated when computing the
 * initial camera matrix from a collection of @ref PlanarView observations.
 */
struct IntrinsicsEstimateOptions final {
    std::optional<CalibrationBounds> bounds = std::nullopt;  ///< Optional parameter bounds
    bool use_skew = false;                                   ///< Estimate skew parameter
};

/**
 * @brief Result of linear intrinsic estimation
 * @ingroup camera_calibration
 *
 * Contains the estimated camera matrix and the per-view poses recovered
 * from homography decomposition.
 */
struct IntrinsicsEstimateResult final {
    CameraMatrix kmtx;                       ///< Estimated intrinsic matrix
    std::vector<Eigen::Isometry3d> c_se3_t;  ///< Estimated pose of each view
};

/**
 * @brief Estimate camera intrinsics from planar views using a linear method
 * @ingroup camera_calibration
 *
 * Fits a homography for each planar view, extracts the corresponding camera
 * pose and constructs normalized observations. These observations are then
 * used by @ref estimate_intrinsics_linear to compute the camera matrix.
 *
 * @param views      Vector of planar views with pixel measurements
 * @param image_size Image dimensions in pixels (width, height)
 * @param opts       Estimation options (bounds and skew)
 * @return Optional result containing camera matrix and per-view poses
 */
auto estimate_intrinsics(const std::vector<PlanarView>& views, const Eigen::Vector2i& image_size,
                         const IntrinsicsEstimateOptions& opts = {})
    -> std::optional<IntrinsicsEstimateResult>;

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
auto estimate_intrinsics_linear_iterative(const std::vector<Observation<double>>& observations,
                                          int num_radial,
                                          int max_iterations = k_default_max_iterations,
                                          bool use_skew = false)
    -> std::optional<PinholeCamera<BrownConradyd>>;

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

auto optimize_intrinsics_semidlt(const std::vector<PlanarView>& views,
                                 const CameraMatrix& initial_guess,
                                 const IntrinsicsOptions& opts = {})
    -> IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>>;

template <camera_model CameraT>
auto optimize_intrinsics(const std::vector<PlanarView>& views, const CameraT& init_camera,
                         std::vector<Eigen::Isometry3d> init_c_se3_t,
                         const IntrinsicsOptions& opts = {})
    -> IntrinsicsOptimizationResult<CameraT>;

}  // namespace calib
