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
#include <utility>
#include <vector>

#include "calib/charuco_intrinsics_utils.h"
#include "calib/intrinsics.h"
#include "calib/planarpose.h"
#include "calib/ransac.h"

using namespace calib;
using namespace calib::charuco;

namespace calib::charuco {

[[nodiscard]] auto load_config(const std::filesystem::path& path) -> CharucoCalibrationConfig {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open config: " + path.string());
    }

    nlohmann::json json_cfg;
    stream >> json_cfg;

    CharucoCalibrationConfig cfg;
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

}  // namespace calib::charuco

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

        CharucoIntrinsicCalibrationFacade facade;

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

            auto result = facade.calibrate(cfg, cam_cfg, detections, features_path);
            print_calibration_summary(std::cout, cam_cfg, result.outputs);

            camera_results.push_back(std::move(result.report));

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
