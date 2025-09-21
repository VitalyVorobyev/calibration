/**
 * @file planar_intrinsics_utils.h
 * @brief Shared data structures and helpers for planar-target intrinsic calibration.
 */

#pragma once

// std
#include <array>
#include <filesystem>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "calib/estimation/intrinsics.h"
#include "calib/estimation/ransac.h"
#include "calib/models/pinhole.h"

namespace calib::planar {

/**
 * @brief Metadata describing a calibration session.
 */
struct SessionConfig {
    std::string id;
    std::string description;
};

/**
 * @brief Configuration for RANSAC-based homography estimation.
 */
struct HomographyRansacConfig {
    int max_iters = 2000;
    double thresh = 1.5;
    int min_inliers = 30;
    double confidence = 0.99;
};

/**
 * @brief Options that control planar intrinsic calibration behaviour.
 */
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

/**
 * @brief Static camera properties required to initialise a calibration run.
 */
struct CameraConfig {
    std::string camera_id;
    std::string model = "pinhole_brown_conrady";
    std::optional<std::array<int, 2>> image_size;
};

/**
 * @brief High-level calibration description consumed by the facade.
 */
struct PlanarCalibrationConfig {
    SessionConfig session;
    std::string algorithm = "planar";
    IntrinsicCalibrationOptions options;
    std::vector<CameraConfig> cameras;
};

/**
 * @brief Observation of a single planar target point in both image and board space.
 */
struct PlanarTargetPoint {
    double x = 0.0;
    double y = 0.0;
    int id = -1;
    double local_x = 0.0;
    double local_y = 0.0;
    double local_z = 0.0;
};

/**
 * @brief Detection results for one image of the planar target.
 */
struct PlanarImageDetections {
    std::string file;
    int count = 0;
    std::vector<PlanarTargetPoint> points;
};

/**
 * @brief Collection of planar detections along with detector metadata.
 */
struct PlanarDetections {
    std::string image_directory;
    std::string feature_type;
    std::string algo_version;
    std::string params_hash;
    std::string sensor_id;
    std::set<std::string> tags;
    nlohmann::json metadata = nlohmann::json::object();
    std::filesystem::path source_file;
    std::vector<PlanarImageDetections> images;
};

/**
 * @brief Description of a view that passed the filtering stage.
 */
struct ActiveView {
    std::string source_image;
    std::size_t corner_count = 0;
};

/**
 * @brief Results produced by the planar intrinsic calibration pipeline.
 */
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

/**
 * @brief Count the number of non-overlapping occurrences of a substring.
 */
[[nodiscard]] auto count_occurrences(std::string_view text, std::string_view needle) -> std::size_t;

/**
 * @brief Compute the board centre either from overrides or from observed points.
 */
[[nodiscard]] auto determine_point_center(const PlanarDetections& detections,
                                          const IntrinsicCalibrationOptions& opts)
    -> std::array<double, 2>;

/**
 * @brief Convert detector results into planar views suitable for calibration.
 */
[[nodiscard]] auto collect_planar_views(const PlanarDetections& detections,
                                        const IntrinsicCalibrationOptions& opts,
                                        const std::array<double, 2>& point_center,
                                        std::vector<ActiveView>& views) -> std::vector<PlanarView>;

/**
 * @brief Translate the user-facing RANSAC configuration to library options.
 */
[[nodiscard]] auto build_ransac_options(const HomographyRansacConfig& cfg) -> RansacOptions;

/**
 * @brief Serialise distortion coefficients to JSON.
 */
[[nodiscard]] auto distortion_to_json(const Eigen::VectorXd& coeffs) -> nlohmann::json;

/**
 * @brief Serialise a camera matrix to JSON.
 */
[[nodiscard]] auto camera_matrix_to_json(const CameraMatrix& k) -> nlohmann::json;

/**
 * @brief Aggregate per-view RMS errors into a global RMS value.
 */
[[nodiscard]] auto compute_global_rms(const CalibrationOutputs& out) -> double;

/**
 * @brief Construct the calibration report JSON payload.
 */
[[nodiscard]] auto build_output_json(const PlanarCalibrationConfig& cfg,
                                     const CameraConfig& cam_cfg,
                                     const PlanarDetections& detections,
                                     const CalibrationOutputs& outputs,
                                     const std::filesystem::path& features_path) -> nlohmann::json;

/**
 * @brief Combined result containing both raw outputs and JSON report.
 */
struct CalibrationRunResult {
    CalibrationOutputs outputs;
    nlohmann::json report;
};

/**
 * @brief Facade that orchestrates planar intrinsic calibration end-to-end.
 */
class PlanarIntrinsicCalibrationFacade {
  public:
    auto calibrate(const PlanarCalibrationConfig& cfg, const CameraConfig& cam_cfg,
                   const PlanarDetections& detections,
                   const std::filesystem::path& features_path) const -> CalibrationRunResult;
};

/**
 * @brief Emit a human-readable summary of calibration results.
 */
void print_calibration_summary(std::ostream& out, const CameraConfig& cam_cfg,
                               const CalibrationOutputs& outputs);

/**
 * @brief Load a planar calibration configuration from JSON.
 */
[[nodiscard]] auto load_calibration_config(const std::filesystem::path& path)
    -> PlanarCalibrationConfig;

/**
 * @brief Load planar target detections from JSON.
 */
[[nodiscard]] auto load_planar_observations(const std::filesystem::path& path,
                                            std::optional<std::string> sensor_filter = std::nullopt)
    -> PlanarDetections;

[[nodiscard]] auto validate_planar_dataset(const nlohmann::json& dataset,
                                           std::string* error_message = nullptr) -> bool;

[[nodiscard]] auto convert_legacy_planar_features(const nlohmann::json& legacy,
                                                  const std::string& sensor_id_hint = "cam0")
    -> nlohmann::json;

}  // namespace calib::planar
