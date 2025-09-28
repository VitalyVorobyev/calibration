#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "calib/estimation/optim/optimize.h"

namespace calib::planar {

struct SessionConfig {
    std::string id;
    std::string description;
};

struct HomographyRansacConfig {
    int max_iters = 2000;
    double thresh = 1.5;
    int min_inliers = 30;
    double confidence = 0.99;
};

struct IntrinsicCalibrationOptions {
    std::size_t min_corners_per_view = 80;
    bool refine = true;
    bool optimize_skew = false;
    int num_radial = 3;
    double huber_delta = 2.0;
    int max_iterations = 200;
    double epsilon = OptimOptions::k_default_epsilon;
    bool verbose = false;
    double point_scale = 1.0;
    bool auto_center = true;
    std::optional<std::array<double, 2>> point_center_override;
    std::vector<int> fixed_distortion_indices;
    std::vector<double> fixed_distortion_values;
    std::optional<HomographyRansacConfig> homography_ransac;
};

struct CameraConfig {
    std::string camera_id;
    std::string model = "pinhole_brown_conrady";
    std::optional<std::array<int, 2>> image_size;
};

struct PlanarCalibrationConfig {
    SessionConfig session;
    std::string algorithm = "planar";
    IntrinsicCalibrationOptions options;
    std::vector<CameraConfig> cameras;
};

}  // namespace calib::planar
