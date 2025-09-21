#include "calib/reports/planar_intrinsics.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>


#include <algorithm>
namespace calib::planar {

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

auto build_planar_intrinsics_report(const PlanarCalibrationConfig& cfg, const CameraConfig& cam_cfg,
                                    const PlanarDetections& detections,
                                    const CalibrationOutputs& outputs,
                                    const std::filesystem::path& /*features_path*/)
    -> nlohmann::json {
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
    }
    if (!cfg.options.fixed_distortion_values.empty()) {
        options_json["fixed_distortion_values"] = cfg.options.fixed_distortion_values;
    }
    if (cfg.options.homography_ransac.has_value()) {
        const auto& r = *cfg.options.homography_ransac;
        options_json["homography_ransac"] = {{"max_iters", r.max_iters},
                                              {"thresh", r.thresh},
                                              {"min_inliers", r.min_inliers},
                                              {"confidence", r.confidence}};
    }

    nlohmann::json detector_json;
    if (!detections.metadata.is_null() && detections.metadata.contains("detector")) {
        detector_json = detections.metadata.at("detector");
    } else {
        detector_json = nlohmann::json::object();
    }

    nlohmann::json camera_json;
    camera_json["camera_id"] = cam_cfg.camera_id;
    camera_json["model"] = cam_cfg.model;
    if (cam_cfg.image_size.has_value()) {
        camera_json["image_size"] = {*cam_cfg.image_size};
    }
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
                                    {"detector", detector_json},
                                    {"cameras", nlohmann::json::array({camera_json})}};

    return nlohmann::json{{"session", session_json},
                          {"calibrations", nlohmann::json::array({calibration_json})}};
}

}  // namespace calib::planar

