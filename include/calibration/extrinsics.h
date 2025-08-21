#pragma once

// std
#include <vector>
#include <string>

// eigen
#include <Eigen/Geometry>

#include "calibration/planarpose.h"
#include "calibration/intrinsics.h"

namespace vitavision {

struct ExtrinsicPlanarView {
    // observations[camera][feature]
    std::vector<std::vector<PlanarObservation>> observations;
};

struct ExtrinsicOptimizationResult {
    std::vector<Eigen::Affine3d> camera_poses;  // reference->camera
    std::vector<Eigen::Affine3d> target_poses;  // target->reference
    double reprojection_error = 0.0;           // RMS pixel error
    std::string summary;                       // Solver brief report
};

ExtrinsicOptimizationResult optimize_extrinsic_poses(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<CameraMatrix>& intrinsics,
    const std::vector<Eigen::VectorXd>& distortions,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose = false);

} // namespace vitavision

