/**
 * @file homography.h
 * @brief Homography estimation and refinement algorithms
 * @ingroup geometric_transforms
 *
 * This file provides comprehensive homography computation functionality including:
 * - Direct Linear Transform (DLT) for initial estimation
 * - Bundle adjustment for non-linear refinement
 * - Robust estimation with outlier rejection
 * - Uncertainty quantification through covariance estimation
 */

#pragma once

// std
#include <vector>
#include <optional>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/optimize.h"

namespace calib {

struct HomographyResult final {
    bool success{false};
    Eigen::Matrix3d hmtx = Eigen::Matrix3d::Identity();
    std::vector<int> inliers;      // indices of inlier correspondences
    double symmetric_rms_px{0.0};  // symmetric transfer RMS in pixels
};

struct RansacOptions final {
    int ransac_max_iters = 1000;
    double ransac_thresh_px = 2.0;  // symmetric transfer threshold
    int ransac_min_inliers = 12;
};

/**
 * @brief Estimates a 3x3 homography matrix using the Direct Linear Transform (DLT) algorithm.
 *
 * Given two sets of corresponding 2D points, this function computes the homography matrix
 * that best maps the source points to the destination points in a least-squares sense.
 *
 * @param src A vector of 2D source points (Eigen::Vector2d).
 * @param dst A vector of 2D destination points (Eigen::Vector2d), corresponding to src.
 * @return Eigen::Matrix3d The estimated 3x3 homography matrix such that dst ≈ H * src.
 *
 * @note Both src and dst must contain at least 4 points and have the same size.
 */
auto estimate_homography(const std::vector<Eigen::Vector2d>& src,
                         const std::vector<Eigen::Vector2d>& dst,
                         std::optional<RansacOptions> ransac_opts) -> HomographyResult;

struct HomographyOptions final : public OptimOptions {};

struct OptimizeHomographyResult final : OptimResult {
    Eigen::Matrix3d homography;
};

/**
 * @brief Optimizes the homography transformation between two sets of 2D points.
 *
 * This function computes the optimal homography matrix that maps the source points (`src`)
 * to the destination points (`dst`) using a specified optimization strategy.
 *
 * @param src A vector of 2D points (Eigen::Vector2d) representing the source coordinates.
 * @param dst A vector of 2D points (Eigen::Vector2d) representing the destination coordinates.
 * @param options Optional optimization parameters (default is OptimOptions()).
 * @return OptimizeHomographyResult The result of the optimization, including the estimated
 * homography matrix and optimization status.
 *
 * @note The input vectors `src` and `dst` must have the same size and contain at least four points.
 */
auto optimize_homography(const std::vector<Eigen::Vector2d>& src,
                         const std::vector<Eigen::Vector2d>& dst, const Eigen::Matrix3d& init_h,
                         const HomographyOptions& options = {}) -> OptimizeHomographyResult;

}  // namespace calib
