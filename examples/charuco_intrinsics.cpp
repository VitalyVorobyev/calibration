#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "calib/charuco_intrinsics_utils.h"
#include "calib/intrinsics.h"
#include "calib/planarpose.h"
#include "calib/ransac.h"

using namespace calib;
using namespace calib::charuco;

namespace {

class StreamCapture final {
  public:
    explicit StreamCapture(std::ostream& stream)
        : stream_(stream), old_buf_(stream.rdbuf(buffer_.rdbuf())) {}
    StreamCapture(const StreamCapture&) = delete;
    StreamCapture& operator=(const StreamCapture&) = delete;
    ~StreamCapture() { stream_.rdbuf(old_buf_); }

    [[nodiscard]] auto str() const -> std::string { return buffer_.str(); }

  private:
    std::ostream& stream_;
    std::ostringstream buffer_;
    std::streambuf* old_buf_;
};

[[nodiscard]] auto load_config(const std::filesystem::path& path) -> ExampleConfig {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open config: " + path.string());
    }

    nlohmann::json json_cfg;
    stream >> json_cfg;

    ExampleConfig cfg;
    const auto& session = json_cfg.at("session");
    cfg.session.id = session.value("id", "charuco_session");
    cfg.session.description = session.value("description", "");

    const auto& calib = json_cfg.at("calibration");
    const auto type = calib.value("type", "intrinsics");
    if (type != "intrinsics") {
        throw std::runtime_error("Example only supports intrinsics calibration, got: " + type);
    }
    cfg.algorithm = calib.value("algorithm", "charuco_planar");

    const auto& opts_json = calib.value("options", nlohmann::json::object());
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

    const auto& cams_json = json_cfg.at("cameras");
    if (!cams_json.is_array() || cams_json.empty()) {
        throw std::runtime_error("Config must list at least one camera under 'cameras'.");
    }

    cfg.cameras.reserve(cams_json.size());
    for (const auto& cam_json : cams_json) {
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

[[nodiscard]] auto load_charuco_features(const std::filesystem::path& path) -> CharucoDetections {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open features JSON: " + path.string());
    }

    nlohmann::json json_data;
    stream >> json_data;

    CharucoDetections detections;
    detections.image_directory = json_data.value("image_directory", "");
    detections.feature_type = json_data.value("feature_type", "");
    detections.algo_version = json_data.value("algo_version", "");
    detections.params_hash = json_data.value("params_hash", "");

    if (detections.feature_type != "charuco") {
        throw std::runtime_error("Expected feature_type 'charuco', got: '" +
                                 detections.feature_type + "'");
    }

    const auto& images_json = json_data.at("images");
    detections.images.reserve(images_json.size());
    for (const auto& img_json : images_json) {
        CharucoImageDetections img;
        img.count = img_json.value("count", 0);
        img.file = img_json.value("file", "");
        const auto& points_json = img_json.at("points");
        img.points.reserve(points_json.size());
        for (const auto& pt_json : points_json) {
            CharucoPoint pt;
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

[[nodiscard]] auto calibrate_camera(const ExampleConfig& config, const CameraConfig& cam_cfg,
                                    const CharucoDetections& detections) -> CalibrationOutputs {
    CalibrationOutputs output;
    output.total_input_views = detections.images.size();
    output.min_corner_threshold = config.options.min_corners_per_view;
    output.point_scale = config.options.point_scale;
    const auto point_center = determine_point_center(detections, config.options);
    output.point_center = point_center;

    std::vector<ActiveView> active_views;
    auto planar_views =
        collect_planar_views(detections, config.options, point_center, active_views);
    output.accepted_views = planar_views.size();

    if (planar_views.size() < 4) {
        std::ostringstream oss;
        oss << "Need at least 4 views with >= " << config.options.min_corners_per_view
            << " corners. Only " << planar_views.size() << " usable views.";
        throw std::runtime_error(oss.str());
    }

    IntrinsicsEstimateOptions estimate_opts;
    estimate_opts.use_skew = false;
    if (config.options.homography_ransac.has_value()) {
        estimate_opts.homography_ransac = build_ransac_options(*config.options.homography_ransac);
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
    refine_opts.optimize_skew = config.options.optimize_skew;
    refine_opts.num_radial = config.options.num_radial;
    refine_opts.huber_delta = config.options.huber_delta;
    refine_opts.max_iterations = config.options.max_iterations;
    refine_opts.epsilon = config.options.epsilon;
    refine_opts.verbose = config.options.verbose;
    refine_opts.bounds = bounds;
    refine_opts.fixed_distortion_indices = config.options.fixed_distortion_indices;
    refine_opts.fixed_distortion_values = config.options.fixed_distortion_values;

    IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>> refine;
    if (config.options.refine) {
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
    return output;
}

void log_summary(const CameraConfig& cam_cfg, const CalibrationOutputs& outputs) {
    std::cout << "== Camera " << cam_cfg.camera_id << " ==\n";
    std::cout << "Point scale applied to board coordinates: " << outputs.point_scale << '\n';
    std::cout << "Point center removed before scaling: [" << outputs.point_center[0] << ", "
              << outputs.point_center[1] << "]\n";
    if (outputs.invalid_k_warnings > 0 || outputs.pose_warnings > 0) {
        std::cout << "Linear stage warnings: " << outputs.invalid_k_warnings
                  << " invalid camera matrices, " << outputs.pose_warnings
                  << " homography decompositions\n";
    }
    std::cout << "Initial fx/fy/cx/cy: " << outputs.linear_kmtx.fx << ", " << outputs.linear_kmtx.fy
              << ", " << outputs.linear_kmtx.cx << ", " << outputs.linear_kmtx.cy << '\n';
    const auto& refined = outputs.refine_result.camera;
    std::cout << "Refined fx/fy/cx/cy: " << refined.kmtx.fx << ", " << refined.kmtx.fy << ", "
              << refined.kmtx.cx << ", " << refined.kmtx.cy << '\n';
    std::cout << "Distortion coeffs: " << outputs.refine_result.camera.distortion.coeffs.transpose()
              << '\n';
    std::cout << "Views considered: " << outputs.total_input_views
              << ", after threshold: " << outputs.accepted_views << '\n';
    std::cout << "Per-view RMS (px):";
    for (std::size_t i = 0; i < outputs.refine_result.view_errors.size(); ++i) {
        std::cout << ' ' << outputs.refine_result.view_errors[i];
    }
    std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Intrinsic calibration from ChArUco detections"};

    std::string config_path;
    std::vector<std::string> feature_paths;
    std::string output_path;

    app.add_option("--config", config_path, "Calibration config JSON")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--features", feature_paths, "ChArUco detections JSON (repeat per camera)")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-o,--output", output_path, "Write calibration report to this JSON file");

    CLI11_PARSE(app, argc, argv);

    try {
        const auto cfg = load_config(config_path);
        if (cfg.cameras.size() != feature_paths.size()) {
            if (feature_paths.size() == 1 && cfg.cameras.size() == 1) {
                // ok
            } else {
                std::ostringstream oss;
                oss << "Number of feature files (" << feature_paths.size()
                    << ") does not match cameras in config (" << cfg.cameras.size() << ").";
                throw std::runtime_error(oss.str());
            }
        }

        const std::size_t camera_count = cfg.cameras.size();
        std::vector<nlohmann::json> camera_results;
        camera_results.reserve(camera_count);

        for (std::size_t cam_idx = 0; cam_idx < camera_count; ++cam_idx) {
            const auto& cam_cfg = cfg.cameras[cam_idx];
            const std::filesystem::path features_path =
                feature_paths.size() == 1 ? std::filesystem::path(feature_paths[0])
                                          : std::filesystem::path(feature_paths[cam_idx]);

            std::cerr << "[" << cam_cfg.camera_id << "] Loading detections from " << features_path
                      << '\n';
            const auto detections = load_charuco_features(features_path);
            std::cerr << "[" << cam_cfg.camera_id << "] Found " << detections.images.size()
                      << " image detections" << '\n';

            const auto outputs = calibrate_camera(cfg, cam_cfg, detections);
            log_summary(cam_cfg, outputs);

            const auto json_out =
                build_output_json(cfg, cam_cfg, detections, outputs, features_path);
            camera_results.push_back(json_out);

            if (camera_count > 1) {
                std::cout << std::string(40, '-') << "\n";
            }
        }

        // For now we only support single calibration block per run.
        // camera_results already hold full document for each camera; reuse first when aggregating.
        nlohmann::json final_json;
        if (!camera_results.empty()) {
            final_json = camera_results.front();
            if (camera_results.size() > 1) {
                auto& calib_array = final_json["calibrations"][0]["cameras"];
                calib_array = nlohmann::json::array();
                for (const auto& cam_json : camera_results) {
                    const auto& cameras = cam_json.at("calibrations")[0].at("cameras");
                    calib_array.push_back(cameras[0]);
                }
            }
        }

        if (!output_path.empty()) {
            std::ofstream out(output_path);
            if (!out) {
                throw std::runtime_error("Failed to open output file: " + output_path);
            }
            out << final_json.dump(2) << '\n';
            std::cerr << "Saved calibration report to " << output_path << '\n';
        } else {
            std::cout << final_json.dump(2) << '\n';
        }

    } catch (const std::exception& ex) {
        std::cerr << "Calibration failed: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
