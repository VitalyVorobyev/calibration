#include "calib/charuco_intrinsics_utils.h"

// std
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>

namespace calib::charuco {

namespace {

[[nodiscard]] auto iso_timestamp_utc() -> std::string {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%FT%TZ");
    return oss.str();
}

}  // namespace

auto count_occurrences(std::string_view text, std::string_view needle) -> std::size_t {
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

auto determine_point_center(const CharucoDetections& detections,
                            const IntrinsicsExampleOptions& opts) -> std::array<double, 2> {
    if (opts.point_center_override.has_value()) {
        return opts.point_center_override.value();
    }
    if (!opts.auto_center) {
        return {0.0, 0.0};
    }

    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    bool has_points = false;
    for (const auto& img : detections.images) {
        for (const auto& pt : img.points) {
            min_x = std::min(min_x, pt.local_x);
            max_x = std::max(max_x, pt.local_x);
            min_y = std::min(min_y, pt.local_y);
            max_y = std::max(max_y, pt.local_y);
            has_points = true;
        }
    }
    if (!has_points) {
        return {0.0, 0.0};
    }
    return {0.5 * (min_x + max_x), 0.5 * (min_y + max_y)};
}

auto collect_planar_views(const CharucoDetections& detections, const IntrinsicsExampleOptions& opts,
                          const std::array<double, 2>& point_center, std::vector<ActiveView>& views)
    -> std::vector<PlanarView> {
    std::vector<PlanarView> planar_views;
    views.clear();
    planar_views.reserve(detections.images.size());
    for (const auto& img : detections.images) {
        if (img.points.size() < opts.min_corners_per_view) {
            continue;
        }
        PlanarView view;
        view.reserve(img.points.size());
        for (const auto& pt : img.points) {
            PlanarObservation obs;
            obs.object_xy = Eigen::Vector2d((pt.local_x - point_center[0]) * opts.point_scale,
                                            (pt.local_y - point_center[1]) * opts.point_scale);
            obs.image_uv = Eigen::Vector2d(pt.x, pt.y);
            view.push_back(std::move(obs));
        }
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

auto distortion_to_json(const Eigen::VectorXd& coeffs) -> nlohmann::json {
    nlohmann::json arr = nlohmann::json::array();
    for (int i = 0; i < coeffs.size(); ++i) {
        arr.push_back(coeffs[i]);
    }
    return arr;
}

auto camera_matrix_to_json(const CameraMatrix& k) -> nlohmann::json {
    return nlohmann::json{{"fx", k.fx}, {"fy", k.fy}, {"cx", k.cx}, {"cy", k.cy}, {"skew", k.skew}};
}

auto compute_global_rms(const CalibrationOutputs& out) -> double {
    const auto& refine = out.refine_result;
    if (refine.view_errors.empty()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    std::size_t total_measurements = 0;
    for (std::size_t i = 0; i < refine.view_errors.size(); ++i) {
        const double view_rms = refine.view_errors[i];
        const std::size_t points =
            i < out.active_views.size() ? out.active_views[i].corner_count : 0;
        const std::size_t measurements = points * 2;
        sum_sq += view_rms * view_rms * static_cast<double>(measurements);
        total_measurements += measurements;
    }
    if (total_measurements == 0) {
        return 0.0;
    }
    return std::sqrt(sum_sq / static_cast<double>(total_measurements));
}

auto build_output_json(const ExampleConfig& cfg, const CameraConfig& cam_cfg,
                       const CharucoDetections& detections, const CalibrationOutputs& outputs,
                       const std::filesystem::path& features_path) -> nlohmann::json {
    nlohmann::json session_json{{"id", cfg.session.id},
                                {"description", cfg.session.description},
                                {"timestamp_utc", iso_timestamp_utc()}};

    nlohmann::json options_json{{"min_corners_per_view", cfg.options.min_corners_per_view},
                                {"refine", cfg.options.refine},
                                {"optimize_skew", cfg.options.optimize_skew},
                                {"num_radial", cfg.options.num_radial},
                                {"huber_delta", cfg.options.huber_delta},
                                {"max_iterations", cfg.options.max_iterations},
                                {"epsilon", cfg.options.epsilon},
                                {"point_scale", cfg.options.point_scale},
                                {"auto_center_points", cfg.options.auto_center}};
    if (cfg.options.point_center_override.has_value()) {
        const auto& center = cfg.options.point_center_override.value();
        options_json["point_center"] = {center[0], center[1]};
    }
    if (!cfg.options.fixed_distortion_indices.empty()) {
        options_json["fixed_distortion_indices"] = cfg.options.fixed_distortion_indices;
        if (!cfg.options.fixed_distortion_values.empty()) {
            options_json["fixed_distortion_values"] = cfg.options.fixed_distortion_values;
        }
    }
    if (cfg.options.homography_ransac.has_value()) {
        const auto& r = *cfg.options.homography_ransac;
        options_json["homography_ransac"] = {{"max_iters", r.max_iters},
                                             {"thresh", r.thresh},
                                             {"min_inliers", r.min_inliers},
                                             {"confidence", r.confidence}};
    }

    nlohmann::json camera_json{{"camera_id", cam_cfg.camera_id}, {"model", cam_cfg.model}};
    if (cam_cfg.image_size.has_value()) {
        camera_json["image_size"] = {(*cam_cfg.image_size)[0], (*cam_cfg.image_size)[1]};
    }

    camera_json["input"] = {{"features_file", features_path.string()},
                            {"image_directory", detections.image_directory},
                            {"algo_version", detections.algo_version},
                            {"params_hash", detections.params_hash},
                            {"total_views", outputs.total_input_views},
                            {"views_after_threshold", outputs.accepted_views},
                            {"min_corner_threshold", outputs.min_corner_threshold},
                            {"observations_used", outputs.total_points_used},
                            {"point_center", {outputs.point_center[0], outputs.point_center[1]}},
                            {"point_scale", outputs.point_scale}};

    camera_json["initial_guess"] = {
        {"intrinsics", camera_matrix_to_json(outputs.linear_kmtx)},
        {"used_view_indices", outputs.linear_view_indices},
        {"warning_counts",
         {{"invalid_camera_matrix", outputs.invalid_k_warnings},
          {"homography_decomposition_failures", outputs.pose_warnings}}}};

    const auto global_rms = compute_global_rms(outputs);
    nlohmann::json per_view = nlohmann::json::array();
    for (std::size_t i = 0; i < outputs.active_views.size(); ++i) {
        const auto& view = outputs.active_views[i];
        const double view_rms = i < outputs.refine_result.view_errors.size()
                                    ? outputs.refine_result.view_errors[i]
                                    : 0.0;
        const bool used_in_linear =
            std::find(outputs.linear_view_indices.begin(), outputs.linear_view_indices.end(), i) !=
            outputs.linear_view_indices.end();
        per_view.push_back({{"source_image", view.source_image},
                            {"corner_count", view.corner_count},
                            {"rms_px", view_rms},
                            {"used_in_linear_stage", used_in_linear}});
    }

    nlohmann::json result_json;
    result_json["intrinsics"] = camera_matrix_to_json(outputs.refine_result.camera.kmtx);
    result_json["distortion"] = {
        {"model", cam_cfg.model},
        {"coefficients", distortion_to_json(outputs.refine_result.camera.distortion.coeffs)}};
    result_json["statistics"] = {{"reprojection_rms_px", global_rms}, {"per_view", per_view}};

    camera_json["result"] = result_json;

    nlohmann::json calibration_json{{"type", "intrinsics"},
                                    {"algorithm", cfg.algorithm},
                                    {"options", options_json},
                                    {"cameras", nlohmann::json::array({camera_json})}};

    return nlohmann::json{{"session", session_json},
                          {"calibrations", nlohmann::json::array({calibration_json})}};
}

}  // namespace calib::charuco
