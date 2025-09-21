#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "calib/pipeline/planar_intrinsics.h"
#include "calib/datasets/planar.h"

using namespace calib::planar;

int main(int argc, char** argv) {
    CLI::App app{"Intrinsic calibration from planar target detections"};

    std::string config_path;
    std::vector<std::string> feature_paths;
    std::string output_path;

    app.add_option("--config", config_path, "Calibration config JSON")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--features", feature_paths, "Detections JSON (repeat per camera)")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-o,--output", output_path, "Write calibration report to this JSON file");

    CLI11_PARSE(app, argc, argv);

    try {
        const auto cfg = load_calibration_config(config_path);
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

        PlanarIntrinsicCalibrationFacade facade;

        for (std::size_t cam_idx = 0; cam_idx < camera_count; ++cam_idx) {
            const auto& cam_cfg = cfg.cameras[cam_idx];
            const std::filesystem::path features_path =
                feature_paths.size() == 1 ? std::filesystem::path(feature_paths[0])
                                          : std::filesystem::path(feature_paths[cam_idx]);

            std::cerr << "[" << cam_cfg.camera_id << "] Loading detections from " << features_path
                      << '\n';
            const auto detections = load_planar_dataset(features_path);
            std::cerr << "[" << cam_cfg.camera_id << "] Found " << detections.images.size()
                      << " image detections" << '\n';

            auto result = facade.calibrate(cfg, cam_cfg, detections, features_path);
            print_calibration_summary(std::cout, cam_cfg, result.outputs);

            camera_results.push_back(std::move(result.report));

            if (camera_count > 1) {
                std::cout << std::string(40, '-') << "\n";
            }
        }

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
