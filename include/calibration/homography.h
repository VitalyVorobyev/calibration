/** @brief Homography estimation and refinement */

#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

namespace vitavision {

Eigen::Matrix3d fit_homography(const std::vector<Eigen::Vector2d>& src,
                               const std::vector<Eigen::Vector2d>& dst);

}  // namespace vitavision