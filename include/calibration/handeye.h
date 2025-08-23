#pragma once

#include <vector>
#include <optional>

#include <Eigen/Geometry>

namespace vitavision {

struct HandEyeOptions {
    bool optimize_extrinsics = false;
};

struct HandEyeResult {
    Eigen::Affine3d hand_eye = Eigen::Affine3d::Identity();
    std::vector<Eigen::Affine3d> extrinsics; // reference->camera transforms
    double reprojection_error = 0.0;
    std::string summary;
};

// Simple placeholder API for hand-eye calibration.
// The size of initial_extrinsics must be num_cams-1; the first camera is the reference.
HandEyeResult calibrate_hand_eye(
    const std::vector<Eigen::Affine3d>& initial_extrinsics,
    const HandEyeOptions& options = HandEyeOptions());

} // namespace vitavision

