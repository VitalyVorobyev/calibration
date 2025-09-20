#pragma once

// std
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "calib/intrinsics.h"
#include "calib/pinhole.h"
#include "calib/ransac.h"

namespace calib::charuco {

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

struct IntrinsicsExampleOptions {
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

struct ExampleConfig {
    SessionConfig session;
    std::string algorithm = "charuco_planar";
    IntrinsicsExampleOptions options;
    std::vector<CameraConfig> cameras;
};

struct CharucoPoint {
    double x = 0.0;
    double y = 0.0;
    int id = -1;
    double local_x = 0.0;
    double local_y = 0.0;
    double local_z = 0.0;
};

struct CharucoImageDetections {
    std::string file;
    int count = 0;
    std::vector<CharucoPoint> points;
};

struct CharucoDetections {
    std::string image_directory;
    std::string feature_type;
    std::string algo_version;
    std::string params_hash;
    std::vector<CharucoImageDetections> images;
};

struct ActiveView {
    std::string source_image;
    std::size_t corner_count = 0;
};

struct CalibrationOutputs final {
    CameraMatrix linear_kmtx;
    std::vector<std::size_t> linear_view_indices;
    IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>> refine_result;
    std::vector<ActiveView> active_views;
    std::size_t total_input_views = 0;
    std::size_t accepted_views = 0;
    std::size_t used_views = 0;
    std::size_t total_points_used = 0;
    std::size_t min_corner_threshold = 0;
    double point_scale = 1.0;
    std::array<double, 2> point_center{0.0, 0.0};
    std::size_t invalid_k_warnings = 0;
    std::size_t pose_warnings = 0;
};

[[nodiscard]] auto count_occurrences(std::string_view text, std::string_view needle) -> std::size_t;

[[nodiscard]] auto determine_point_center(const CharucoDetections& detections,
                                          const IntrinsicsExampleOptions& opts)
    -> std::array<double, 2>;

[[nodiscard]] auto collect_planar_views(const CharucoDetections& detections,
                                        const IntrinsicsExampleOptions& opts,
                                        const std::array<double, 2>& point_center,
                                        std::vector<ActiveView>& views) -> std::vector<PlanarView>;

[[nodiscard]] auto build_ransac_options(const HomographyRansacConfig& cfg) -> RansacOptions;

[[nodiscard]] auto distortion_to_json(const Eigen::VectorXd& coeffs) -> nlohmann::json;

[[nodiscard]] auto camera_matrix_to_json(const CameraMatrix& k) -> nlohmann::json;

[[nodiscard]] auto compute_global_rms(const CalibrationOutputs& out) -> double;

[[nodiscard]] auto build_output_json(const ExampleConfig& cfg, const CameraConfig& cam_cfg,
                                     const CharucoDetections& detections,
                                     const CalibrationOutputs& outputs,
                                     const std::filesystem::path& features_path) -> nlohmann::json;

}  // namespace calib::charuco
