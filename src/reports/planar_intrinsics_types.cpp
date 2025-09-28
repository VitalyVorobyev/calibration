#include "calib/reports/planar_intrinsics_types.h"

#include <algorithm>
#include <stdexcept>

namespace calib::planar {
namespace {

[[nodiscard]] auto camera_matrix_to_json_impl(const CameraMatrix& k) -> nlohmann::json {
    return nlohmann::json{{"fx", k.fx}, {"fy", k.fy}, {"cx", k.cx}, {"cy", k.cy}, {"skew", k.skew}};
}

[[nodiscard]] auto camera_matrix_from_json_impl(const nlohmann::json& json) -> CameraMatrix {
    CameraMatrix k;
    json.at("fx").get_to(k.fx);
    json.at("fy").get_to(k.fy);
    json.at("cx").get_to(k.cx);
    json.at("cy").get_to(k.cy);
    if (json.contains("skew")) {
        json.at("skew").get_to(k.skew);
    } else {
        k.skew = 0.0;
    }
    return k;
}

}  // namespace

void to_json(nlohmann::json& json, const PlanarIntrinsicsOptionsReport& report) {
    json = nlohmann::json{{"min_corners_per_view", report.min_corners_per_view},
                          {"refine", report.refine},
                          {"optimize_skew", report.optimize_skew},
                          {"num_radial", report.num_radial},
                          {"huber_delta", report.huber_delta},
                          {"max_iterations", report.max_iterations},
                          {"epsilon", report.epsilon},
                          {"point_scale", report.point_scale},
                          {"auto_center_points", report.auto_center_points}};
    if (report.point_center.has_value()) {
        const auto& center = report.point_center.value();
        json["point_center"] = {center[0], center[1]};
    }
    if (!report.fixed_distortion_indices.empty()) {
        json["fixed_distortion_indices"] = report.fixed_distortion_indices;
    }
    if (!report.fixed_distortion_values.empty()) {
        json["fixed_distortion_values"] = report.fixed_distortion_values;
    }
    if (report.homography_ransac.has_value()) {
        const auto& r = report.homography_ransac.value();
        json["homography_ransac"] = {{"max_iters", r.max_iters},
                                     {"thresh", r.thresh},
                                     {"min_inliers", r.min_inliers},
                                     {"confidence", r.confidence}};
    }
}

void from_json(const nlohmann::json& json, PlanarIntrinsicsOptionsReport& report) {
    json.at("min_corners_per_view").get_to(report.min_corners_per_view);
    json.at("refine").get_to(report.refine);
    json.at("optimize_skew").get_to(report.optimize_skew);
    json.at("num_radial").get_to(report.num_radial);
    json.at("huber_delta").get_to(report.huber_delta);
    json.at("max_iterations").get_to(report.max_iterations);
    json.at("epsilon").get_to(report.epsilon);
    json.at("point_scale").get_to(report.point_scale);
    json.at("auto_center_points").get_to(report.auto_center_points);
    if (json.contains("point_center") && !json.at("point_center").is_null()) {
        const auto& arr = json.at("point_center");
        report.point_center =
            std::array<double, 2>{arr.at(0).get<double>(), arr.at(1).get<double>()};
    } else {
        report.point_center.reset();
    }
    if (json.contains("fixed_distortion_indices")) {
        report.fixed_distortion_indices =
            json.at("fixed_distortion_indices").get<std::vector<int>>();
    } else {
        report.fixed_distortion_indices.clear();
    }
    if (json.contains("fixed_distortion_values")) {
        report.fixed_distortion_values =
            json.at("fixed_distortion_values").get<std::vector<double>>();
    } else {
        report.fixed_distortion_values.clear();
    }
    if (json.contains("homography_ransac")) {
        const auto& r_json = json.at("homography_ransac");
        HomographyRansacConfig config;
        r_json.at("max_iters").get_to(config.max_iters);
        r_json.at("thresh").get_to(config.thresh);
        r_json.at("min_inliers").get_to(config.min_inliers);
        r_json.at("confidence").get_to(config.confidence);
        report.homography_ransac = config;
    } else {
        report.homography_ransac.reset();
    }
}

void to_json(nlohmann::json& json, const InitialGuessReport& report) {
    json = nlohmann::json{{"intrinsics", camera_matrix_to_json_impl(report.intrinsics)},
                          {"used_view_indices", report.used_view_indices},
                          {"warning_counts", report.warning_counts}};
}

void from_json(const nlohmann::json& json, InitialGuessReport& report) {
    report.intrinsics = camera_matrix_from_json_impl(json.at("intrinsics"));
    report.used_view_indices = json.at("used_view_indices").get<std::vector<std::size_t>>();
    report.warning_counts = json.at("warning_counts").get<InitialGuessWarningCounts>();
}

void to_json(nlohmann::json& json, const IntrinsicsResultReport& report) {
    json = nlohmann::json{
        {"intrinsics", camera_matrix_to_json_impl(report.intrinsics)},
        {"distortion",
         {{"model", report.distortion_model}, {"coefficients", report.distortion_coefficients}}},
        {"statistics",
         {{"reprojection_rms_px", report.reprojection_rms_px}, {"per_view", report.per_view}}}};
}

void from_json(const nlohmann::json& json, IntrinsicsResultReport& report) {
    report.intrinsics = camera_matrix_from_json_impl(json.at("intrinsics"));
    const auto& distortion = json.at("distortion");
    distortion.at("model").get_to(report.distortion_model);
    report.distortion_coefficients = distortion.at("coefficients").get<std::vector<double>>();
    const auto& stats = json.at("statistics");
    stats.at("reprojection_rms_px").get_to(report.reprojection_rms_px);
    report.per_view = stats.at("per_view").get<std::vector<PlanarViewReport>>();
}

void to_json(nlohmann::json& json, const CameraReport& report) {
    json = nlohmann::json{{"camera_id", report.camera_id},
                          {"model", report.model},
                          {"initial_guess", report.initial_guess},
                          {"result", report.result}};
    if (report.image_size.has_value()) {
        json["image_size"] = *report.image_size;
    }
}

void from_json(const nlohmann::json& json, CameraReport& report) {
    json.at("camera_id").get_to(report.camera_id);
    json.at("model").get_to(report.model);
    if (json.contains("image_size") && !json.at("image_size").is_null()) {
        const auto& img = json.at("image_size");
        report.image_size = std::array<int, 2>{img.at(0).get<int>(), img.at(1).get<int>()};
    } else {
        report.image_size.reset();
    }
    report.initial_guess = json.at("initial_guess").get<InitialGuessReport>();
    report.result = json.at("result").get<IntrinsicsResultReport>();
}

void to_json(nlohmann::json& json, const CalibrationReport& report) {
    json = nlohmann::json{{"type", report.type},
                          {"algorithm", report.algorithm},
                          {"options", report.options},
                          {"detector", report.detector},
                          {"cameras", report.cameras}};
}

void from_json(const nlohmann::json& json, CalibrationReport& report) {
    json.at("type").get_to(report.type);
    json.at("algorithm").get_to(report.algorithm);
    report.options = json.at("options").get<PlanarIntrinsicsOptionsReport>();
    if (json.contains("detector")) {
        report.detector = json.at("detector");
    } else {
        report.detector = nlohmann::json::object();
    }
    report.cameras = json.at("cameras").get<std::vector<CameraReport>>();
}

}  // namespace calib::planar
