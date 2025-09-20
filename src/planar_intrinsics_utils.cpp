#include "calib/planar_intrinsics_utils.h"

#include "calib/stream_capture.h"

// std
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

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

auto determine_point_center(const PlanarDetections& detections,
                            const IntrinsicCalibrationOptions& opts) -> std::array<double, 2> {
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

auto collect_planar_views(const PlanarDetections& detections,
                          const IntrinsicCalibrationOptions& opts,
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

auto build_output_json(const PlanarCalibrationConfig& cfg, const CameraConfig& cam_cfg,
                       const PlanarDetections& detections, const CalibrationOutputs& outputs,
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
        refine = optimize_intrinsics_semidlt(planar_views, linear.kmtx, refine_opts);
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
    result.report = build_output_json(cfg, cam_cfg, detections, result.outputs, features_path);
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

auto load_calibration_config(const std::filesystem::path& path) -> PlanarCalibrationConfig {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open config: " + path.string());
    }

    nlohmann::json json_cfg;
    stream >> json_cfg;

    PlanarCalibrationConfig cfg;
    const auto& session = json_cfg.at("session");
    cfg.session.id = session.value("id", "planar_session");
    cfg.session.description = session.value("description", "");

    const auto& calibration = json_cfg.at("calibration");
    const auto type = calibration.value("type", "intrinsics");
    if (type != "intrinsics") {
        throw std::runtime_error("Planar calibration example only supports intrinsics, got: " +
                                 type);
    }
    cfg.algorithm = calibration.value("algorithm", cfg.algorithm);

    const auto& opts_json = calibration.value("options", nlohmann::json::object());
    cfg.options.min_corners_per_view =
        opts_json.value("min_corners_per_view", cfg.options.min_corners_per_view);
    cfg.options.refine = opts_json.value("refine", cfg.options.refine);
    cfg.options.optimize_skew = opts_json.value("optimize_skew", cfg.options.optimize_skew);
    cfg.options.num_radial = opts_json.value("num_radial", cfg.options.num_radial);
    cfg.options.huber_delta = opts_json.value("huber_delta", cfg.options.huber_delta);
    cfg.options.max_iterations = opts_json.value("max_iterations", cfg.options.max_iterations);
    cfg.options.epsilon = opts_json.value("epsilon", cfg.options.epsilon);
    cfg.options.verbose = opts_json.value("verbose", cfg.options.verbose);
    cfg.options.point_scale = opts_json.value("point_scale", cfg.options.point_scale);
    cfg.options.auto_center = opts_json.value("auto_center_points", cfg.options.auto_center);
    if (opts_json.contains("point_center")) {
        const auto& arr = opts_json.at("point_center");
        if (!arr.is_array() || arr.size() != 2) {
            throw std::runtime_error("options.point_center must be an array [x, y].");
        }
        cfg.options.point_center_override =
            std::array<double, 2>{arr[0].get<double>(), arr[1].get<double>()};
        cfg.options.auto_center = false;
    }
    if (opts_json.contains("fixed_distortion_indices")) {
        const auto& arr = opts_json.at("fixed_distortion_indices");
        if (!arr.is_array()) {
            throw std::runtime_error("options.fixed_distortion_indices must be an array.");
        }
        cfg.options.fixed_distortion_indices.clear();
        for (const auto& v : arr) {
            cfg.options.fixed_distortion_indices.push_back(v.get<int>());
        }
    }
    if (opts_json.contains("fixed_distortion_values")) {
        const auto& arr = opts_json.at("fixed_distortion_values");
        if (!arr.is_array()) {
            throw std::runtime_error("options.fixed_distortion_values must be an array.");
        }
        cfg.options.fixed_distortion_values.clear();
        for (const auto& v : arr) {
            cfg.options.fixed_distortion_values.push_back(v.get<double>());
        }
    }

    if (opts_json.contains("homography_ransac")) {
        const auto& r = opts_json.at("homography_ransac");
        HomographyRansacConfig ransac_cfg;
        ransac_cfg.max_iters = r.value("max_iters", ransac_cfg.max_iters);
        ransac_cfg.thresh = r.value("thresh", ransac_cfg.thresh);
        ransac_cfg.min_inliers = r.value("min_inliers", ransac_cfg.min_inliers);
        ransac_cfg.confidence = r.value("confidence", ransac_cfg.confidence);
        cfg.options.homography_ransac = ransac_cfg;
    }

    const auto& cameras_json = json_cfg.at("cameras");
    if (!cameras_json.is_array() || cameras_json.empty()) {
        throw std::runtime_error("Config must list at least one camera under 'cameras'.");
    }

    cfg.cameras.clear();
    cfg.cameras.reserve(cameras_json.size());
    for (const auto& cam_json : cameras_json) {
        CameraConfig cam;
        cam.camera_id = cam_json.at("camera_id").get<std::string>();
        cam.model = cam_json.value("model", cam.model);
        if (cam_json.contains("image_size")) {
            const auto& arr = cam_json.at("image_size");
            if (!arr.is_array() || arr.size() != 2) {
                throw std::runtime_error("camera.image_size must be an array of [width, height].");
            }
            cam.image_size = std::array<int, 2>{arr[0].get<int>(), arr[1].get<int>()};
        }
        cfg.cameras.push_back(cam);
    }

    return cfg;
}

auto load_planar_observations(const std::filesystem::path& path) -> PlanarDetections {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open features JSON: " + path.string());
    }

    nlohmann::json json_data;
    stream >> json_data;

    PlanarDetections detections;
    detections.image_directory = json_data.value("image_directory", "");
    detections.feature_type = json_data.value("feature_type", "");
    detections.algo_version = json_data.value("algo_version", "");
    detections.params_hash = json_data.value("params_hash", "");

    if (detections.feature_type.empty()) {
        throw std::runtime_error("Feature file missing 'feature_type'.");
    }

    const auto& images_json = json_data.at("images");
    detections.images.reserve(images_json.size());
    for (const auto& img_json : images_json) {
        PlanarImageDetections img;
        img.count = img_json.value("count", 0);
        img.file = img_json.value("file", "");
        const auto& points_json = img_json.at("points");
        img.points.reserve(points_json.size());
        for (const auto& pt_json : points_json) {
            PlanarTargetPoint pt;
            pt.x = pt_json.value("x", 0.0);
            pt.y = pt_json.value("y", 0.0);
            pt.id = pt_json.value("id", -1);
            pt.local_x = pt_json.value("local_x", 0.0);
            pt.local_y = pt_json.value("local_y", 0.0);
            pt.local_z = pt_json.value("local_z", 0.0);
            img.points.push_back(pt);
        }
        detections.images.push_back(std::move(img));
    }

    return detections;
}

}  // namespace calib::planar
