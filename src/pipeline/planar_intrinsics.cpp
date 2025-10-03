#include "calib/pipeline/planar_intrinsics.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "calib/datasets/planar.h"
#include "calib/estimation/optim/planarpose.h"
#include "calib/io/stream_capture.h"
#include "calib/reports/planar_intrinsics.h"

namespace calib::planar {

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
                          const IntrinsicCalibrationOptions& opts,
                          std::vector<ActiveView>& views)
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

auto build_ransac_options(const HomographyRansacConfig& cfg) -> RansacOptions {
    RansacOptions opts;
    opts.max_iters = cfg.max_iters;
    opts.thresh = cfg.thresh;
    opts.min_inliers = cfg.min_inliers;
    opts.confidence = cfg.confidence;
    return opts;
}

auto PlanarIntrinsicCalibrationFacade::calibrate(const PlanarCalibrationConfig& cfg,
                                                 const CameraConfig& cam_cfg,
                                                 const PlanarDetections& detections,
                                                 const std::filesystem::path& features_path) const
    -> CalibrationRunResult {
    CalibrationOutputs output;
    output.total_input_views = detections.images.size();
    output.min_corner_threshold = cfg.options.min_corners_per_view;
    output.point_scale = cfg.options.point_scale;

    const auto point_center = determine_point_center(detections, cfg.options);
    output.point_center = point_center;

    std::vector<ActiveView> active_views;
    auto planar_views = collect_planar_views(detections, cfg.options, point_center, active_views);
    output.accepted_views = planar_views.size();

    if (planar_views.size() < 4) {
        std::ostringstream oss;
        oss << "Need at least 4 views with >= " << cfg.options.min_corners_per_view
            << " corners. Only " << planar_views.size() << " usable views.";
        throw std::runtime_error(oss.str());
    }

    IntrinsicsEstimateOptions estimate_opts;
    estimate_opts.use_skew = false;
    if (cfg.options.homography_ransac.has_value()) {
        estimate_opts.homography_ransac = build_ransac_options(*cfg.options.homography_ransac);
    }

    std::optional<CalibrationBounds> bounds;
    if (cam_cfg.image_size.has_value()) {
        const double width = static_cast<double>((*cam_cfg.image_size)[0]);
        const double height = static_cast<double>((*cam_cfg.image_size)[1]);
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
        bounds = b;
    }
    estimate_opts.bounds = bounds;

    IntrinsicsEstimateResult linear;
    std::string captured_warnings;
    {
        StreamCapture capture(std::cerr);
        linear = estimate_intrinsics(planar_views, estimate_opts);
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

    IntrinsicsOptions refine_opts;
    refine_opts.optimize_skew = cfg.options.optimize_skew;
    refine_opts.num_radial = cfg.options.num_radial;
    refine_opts.huber_delta = cfg.options.huber_delta;
    refine_opts.max_iterations = cfg.options.max_iterations;
    refine_opts.epsilon = cfg.options.epsilon;
    refine_opts.verbose = cfg.options.verbose;
    refine_opts.bounds = bounds;
    refine_opts.fixed_distortion_indices = cfg.options.fixed_distortion_indices;
    refine_opts.fixed_distortion_values = cfg.options.fixed_distortion_values;

    IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>> refine;
    if (cfg.options.refine) {
        PinholeCamera<BrownConradyd> init_camera(linear.kmtx, Eigen::VectorXd::Zero(5));
        refine = optimize_intrinsics(planar_views, init_camera, refine_opts);
        if (!refine.success) {
            std::cerr << "Warning: Non-linear refinement did not converge. Using linear result."
                      << '\n';
            refine.camera = PinholeCamera<BrownConradyd>(linear.kmtx, Eigen::VectorXd::Zero(5));
        }
    } else {
        refine.success = true;
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

    CalibrationRunResult result;
    result.outputs = std::move(output);
    result.report =
        build_planar_intrinsics_report(cfg, cam_cfg, detections, result.outputs, features_path);
    return result;
}

void print_calibration_summary(std::ostream& out, const CameraConfig& cam_cfg,
                               const CalibrationOutputs& outputs) {
    out << "== Camera " << cam_cfg.camera_id << " ==\n";
    out << "Point scale applied to board coordinates: " << outputs.point_scale << '\n';
    out << "Point center removed before scaling: [" << outputs.point_center[0] << ", "
        << outputs.point_center[1] << "]\n";
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

auto load_calibration_config_impl(const std::filesystem::path& path) -> PlanarCalibrationConfig {
    std::ifstream stream(path);
    nlohmann::json json_cfg;
    stream >> json_cfg;

    PlanarCalibrationConfig cfg = json_cfg;
    return cfg;
}

auto load_calibration_config(const std::filesystem::path& path) -> std::optional<PlanarCalibrationConfig> {
    try {
        return load_calibration_config_impl(path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load calibration config from " << path << ": " << e.what()
                  << std::endl;
        return std::nullopt;
    }
}

}  // namespace calib::planar
