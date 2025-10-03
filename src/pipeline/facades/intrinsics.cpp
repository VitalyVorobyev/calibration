#include "calib/pipeline/facades/intrinsics.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "calib/estimation/optim/planarpose.h"
#include "calib/io/stream_capture.h"
#include "calib/pipeline/dataset.h"

namespace calib::pipeline {

namespace {

[[nodiscard]] auto count_occurrences(std::string_view text, std::string_view needle)
    -> std::size_t {
    if (needle.empty()) {
        return 0;
    }
    std::size_t count = 0;
    std::size_t pos = 0;
    while (true) {
        pos = text.find(needle, pos);
        if (pos == std::string_view::npos) {
            break;
        }
        ++count;
        pos += needle.size();
    }
    return count;
}

}  // namespace

auto collect_planar_views(const PlanarDetections& detections,
                          const IntrinsicCalibrationOptions& opts, std::vector<ActiveView>& views)
    -> std::vector<PlanarView> {
    std::vector<PlanarView> planar_views;
    views.clear();
    planar_views.reserve(detections.images.size());
    for (const auto& img : detections.images) {
        if (img.points.size() < opts.min_corners_per_view) {
            continue;
        }
        PlanarView view(img.points.size());
        std::transform(img.points.begin(), img.points.end(), view.begin(), [&](const auto& pt) {
            PlanarObservation obs;
            obs.object_xy = Eigen::Vector2d(pt.local_x, pt.local_y);
            obs.image_uv = Eigen::Vector2d(pt.x, pt.y);
            return obs;
        });
        views.push_back({img.file, view.size()});
        planar_views.push_back(std::move(view));
    }
    return planar_views;
}

auto bounds_from_image_size(const std::array<int, 2>& image_size) -> CalibrationBounds {
    const double width = static_cast<double>(image_size[0]);
    const double height = static_cast<double>(image_size[1]);
    const double short_side = std::min(width, height);
    const double long_side = std::max(width, height);

    CalibrationBounds b;
    b.fx_min = b.fy_min = std::max(1.0, 0.25 * short_side);
    b.fx_max = b.fy_max = std::numeric_limits<double>::max();
    b.cx_min = 0.05 * width;
    b.cx_max = 0.95 * width;
    b.cy_min = 0.05 * height;
    b.cy_max = 0.95 * height;
    const double skew_limit = 0.05 * long_side;
    b.skew_min = -skew_limit;
    b.skew_max = skew_limit;
    return b;
}

auto PlanarIntrinsicCalibrationFacade::calibrate(const IntrinsicCalibrationConfig& cfg,
                                                 const CameraConfig& cam_cfg,
                                                 const PlanarDetections& detections) const
    -> IntrinsicCalibrationOutputs {
    IntrinsicCalibrationOutputs output;
    output.total_input_views = detections.images.size();
    output.min_corner_threshold = cfg.options.min_corners_per_view;

    std::vector<ActiveView> active_views;
    auto planar_views = collect_planar_views(detections, cfg.options, active_views);
    output.accepted_views = planar_views.size();

    if (planar_views.size() < 4) {
        std::ostringstream oss;
        oss << "Need at least 4 views with >= " << cfg.options.min_corners_per_view
            << " corners. Only " << planar_views.size() << " usable views.";
        throw std::runtime_error(oss.str());
    }

    IntrinsicsEstimateResult linear;
    std::string captured_warnings;
    {
        StreamCapture capture(std::cerr);
        linear = estimate_intrinsics(planar_views, cfg.options.estim_options);
        captured_warnings = capture.str();
    }
    output.invalid_k_warnings = count_occurrences(captured_warnings, "Invalid camera matrix K");
    output.pose_warnings = count_occurrences(captured_warnings, "Homography decomposition failed");
    if (output.invalid_k_warnings > 0 || output.pose_warnings > 0) {
        std::cerr << "[" << cam_cfg.camera_id
                  << "] Linear stage warnings: " << output.invalid_k_warnings
                  << " invalid camera matrices, " << output.pose_warnings
                  << " decomposition failures" << '\n';
    }
    if (!linear.success) {
        throw std::runtime_error("Linear intrinsic estimation failed to converge.");
    }

    std::vector<std::size_t> linear_view_indices;
    linear_view_indices.reserve(linear.views.size());
    for (const auto& v : linear.views) {
        linear_view_indices.push_back(v.view_index);
    }

    IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>> refine;
    if (cfg.options.refine) {
        // Estimate initial poses for each view
        std::vector<Eigen::Isometry3d> init_c_se3_t(planar_views.size());
        std::transform(planar_views.begin(), planar_views.end(), init_c_se3_t.begin(),
                       [&](const auto& view) { return estimate_planar_pose(view, linear.kmtx); });

        PinholeCamera<BrownConradyd> init_camera(linear.kmtx, Eigen::VectorXd::Zero(5));
        refine =
            optimize_intrinsics(planar_views, init_camera, init_c_se3_t, cfg.options.optim_options);
        if (!refine.core.success) {
            std::cerr << "Warning: Non-linear refinement did not converge. Using linear result."
                      << '\n';
            refine.camera = PinholeCamera<BrownConradyd>(linear.kmtx, Eigen::VectorXd::Zero(5));
        }
    } else {
        refine.core.success = true;
        refine.camera = PinholeCamera<BrownConradyd>(linear.kmtx, Eigen::VectorXd::Zero(5));
    }

    if (refine.camera.distortion.coeffs.size() == 0) {
        refine.camera.distortion.coeffs = Eigen::VectorXd::Zero(5);
    }

    output.linear_kmtx = linear.kmtx;
    output.linear_view_indices = std::move(linear_view_indices);
    output.refine_result = std::move(refine);
    output.active_views = std::move(active_views);
    output.used_views = planar_views.size();
    output.total_points_used = 0;
    for (const auto& v : output.active_views) {
        output.total_points_used += v.corner_count;
    }

    return output;
}

void print_calibration_summary(std::ostream& out, const CameraConfig& cam_cfg,
                               const IntrinsicCalibrationOutputs& outputs) {
    out << "== Camera " << cam_cfg.camera_id << " ==\n";
    if (outputs.invalid_k_warnings > 0 || outputs.pose_warnings > 0) {
        out << "Linear stage warnings: " << outputs.invalid_k_warnings
            << " invalid camera matrices, " << outputs.pose_warnings
            << " homography decompositions\n";
    }
    out << "Initial fx/fy/cx/cy: " << outputs.linear_kmtx.fx << ", " << outputs.linear_kmtx.fy
        << ", " << outputs.linear_kmtx.cx << ", " << outputs.linear_kmtx.cy << '\n';
    const auto& refined = outputs.refine_result.camera;
    out << "Refined fx/fy/cx/cy: " << refined.kmtx.fx << ", " << refined.kmtx.fy << ", "
        << refined.kmtx.cx << ", " << refined.kmtx.cy << '\n';
    out << "Distortion coeffs: " << outputs.refine_result.camera.distortion.coeffs.transpose()
        << '\n';
    out << "Views considered: " << outputs.total_input_views
        << ", after threshold: " << outputs.accepted_views << '\n';
    out << "Per-view RMS (px):";
    for (double err : outputs.refine_result.view_errors) {
        out << ' ' << err;
    }
    out << "\n";
}

auto load_calibration_config_impl(const std::filesystem::path& path) -> IntrinsicCalibrationConfig {
    std::ifstream stream(path);
    nlohmann::json json_cfg;
    stream >> json_cfg;
    IntrinsicCalibrationConfig cfg = json_cfg;
    return cfg;
}

auto load_calibration_config(const std::filesystem::path& path)
    -> std::optional<IntrinsicCalibrationConfig> {
    try {
        return load_calibration_config_impl(path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load calibration config from " << path << ": " << e.what()
                  << std::endl;
        return std::nullopt;
    }
}

}  // namespace calib::pipeline
