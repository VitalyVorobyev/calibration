#pragma once

#include <array>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "calib/datasets/planar.h"
#include "calib/estimation/common/ransac.h"
#include "calib/estimation/linear/planarpose.h"
#include "calib/estimation/optim/intrinsics.h"
#include "calib/models/pinhole.h"
#include "calib/reports/planar_intrinsics_types.h"

namespace calib::planar {

struct ActiveView final {
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
    PlanarIntrinsicsReport report;
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
