#pragma once

#include <array>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "calib/io/serialization.h"
#include "calib/pipeline/dataset.h"
#include "calib/estimation/common/ransac.h"
#include "calib/estimation/linear/planarpose.h"
#include "calib/estimation/optim/intrinsics.h"
#include "calib/models/pinhole.h"

namespace calib::pipeline {

using calib::from_json;
using calib::to_json;

struct IntrinsicCalibrationOptions final {
    IntrinsicsOptimOptions optim_options;
    IntrinsicsEstimOptions estim_options;
    std::size_t min_corners_per_view = 80;
    bool refine = true;
};

struct CameraConfig final {
    std::string camera_id;
    std::string model = "pinhole_brown_conrady";
    std::optional<std::array<int, 2>> image_size;
};

// TODO: make it configurable
[[nodiscard]] auto bounds_from_image_size(const std::array<int, 2>& image_size)
    -> CalibrationBounds;

struct IntrinsicCalibrationConfig final {
    std::string algorithm = "planar";
    IntrinsicCalibrationOptions options;
    std::vector<CameraConfig> cameras;
};

struct ActiveView final {
    std::string source_image;
    std::size_t corner_count = 0;
};

struct IntrinsicCalibrationOutputs final {
    CameraMatrix linear_kmtx;
    std::vector<std::size_t> linear_view_indices;
    IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>> refine_result;
    std::vector<ActiveView> active_views;
    std::size_t total_input_views = 0;
    std::size_t accepted_views = 0;
    std::size_t used_views = 0;
    std::size_t total_points_used = 0;
    std::size_t min_corner_threshold = 0;
    std::size_t invalid_k_warnings = 0;
    std::size_t pose_warnings = 0;
};

class PlanarIntrinsicCalibrationFacade {
  public:
    [[nodiscard]] auto calibrate(const IntrinsicCalibrationConfig& cfg, const CameraConfig& cam_cfg,
                                 const PlanarDetections& detections,
                                 const std::filesystem::path& features_path) const
        -> IntrinsicCalibrationOutputs;
};

[[nodiscard]] auto collect_planar_views(const PlanarDetections& detections,
                                        const IntrinsicCalibrationOptions& opts,
                                        std::vector<ActiveView>& views) -> std::vector<PlanarView>;

[[nodiscard]] auto load_calibration_config(const std::filesystem::path& path)
    -> std::optional<IntrinsicCalibrationConfig>;

void print_calibration_summary(std::ostream& out, const CameraConfig& cam_cfg,
                               const IntrinsicCalibrationOutputs& outputs);


static_assert(serializable_aggregate<IntrinsicCalibrationOptions>);
static_assert(serializable_aggregate<CameraConfig>);
static_assert(serializable_aggregate<IntrinsicCalibrationConfig>);
static_assert(serializable_aggregate<IntrinsicCalibrationOutputs>);

}  // namespace calib::pipeline
