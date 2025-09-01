/** @brief Homography estimation and refinement */

#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include "calib/optimize.h"

namespace calib {

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
auto estimate_homography_dlt(const std::vector<Eigen::Vector2d>& src,
                            const std::vector<Eigen::Vector2d>& dst) -> Eigen::Matrix3d;

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
                        const std::vector<Eigen::Vector2d>& dst,
                        const Eigen::Matrix3d& init_h,
                        const HomographyOptions& options = {}) -> OptimizeHomographyResult;

}  // namespace calib
