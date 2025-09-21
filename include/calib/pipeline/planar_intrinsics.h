#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "calib/datasets/planar.h"
#include "calib/estimation/intrinsics.h"
#include "calib/estimation/planarpose.h"
#include "calib/estimation/ransac.h"
#include "calib/models/pinhole.h"

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

struct CalibrationRunResult {
    CalibrationOutputs outputs;
    nlohmann::json report;
};

[[nodiscard]] auto determine_point_center(const PlanarDetections& detections,
                                          const IntrinsicCalibrationOptions& opts)
    -> std::array<double, 2>;

class PlanarIntrinsicCalibrationFacade {
  public:
    [[nodiscard]] auto calibrate(const PlanarCalibrationConfig& cfg, const CameraConfig& cam_cfg,
                                 const PlanarDetections& detections,
                                 const std::filesystem::path& features_path) const
        -> CalibrationRunResult;
};

[[nodiscard]] auto collect_planar_views(const PlanarDetections& detections,
                                        const IntrinsicCalibrationOptions& opts,
                                        const std::array<double, 2>& point_center,
                                        std::vector<ActiveView>& views) -> std::vector<PlanarView>;

[[nodiscard]] auto build_ransac_options(const HomographyRansacConfig& cfg) -> RansacOptions;

[[nodiscard]] auto load_calibration_config(const std::filesystem::path& path)
    -> PlanarCalibrationConfig;

void print_calibration_summary(std::ostream& out, const CameraConfig& cam_cfg,
                               const CalibrationOutputs& outputs);

}  // namespace calib::planar

