#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "calib/io/serialization.h"
#include "calib/pipeline/extrinsics.h"
#include "calib/pipeline/loaders.h"
#include "calib/pipeline/pipeline.h"
#include "calib/pipeline/stages.h"

namespace {

[[nodiscard]] auto resolve_path(const std::filesystem::path& base, const std::string& candidate)
    -> std::filesystem::path {
    const std::filesystem::path path(candidate);
    if (path.is_absolute()) {
        return path;
    }
    return base / path;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Planar intrinsics and extrinsics calibration example (stereo or multicam)"};

    std::string input_path;
    std::string output_path = "artifacts.json";
    bool verbose = false;

    app.add_option("--input", input_path, "Pipeline input configuration JSON")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--output", output_path, "Path to write calibration artifacts JSON");
    app.add_flag("-v,--verbose", verbose, "Print pipeline stage transitions");

    CLI11_PARSE(app, argc, argv);

    try {
        std::ifstream input_stream(input_path);
        if (!input_stream) {
            throw std::runtime_error("Failed to open input file: " + input_path);
        }

        nlohmann::json config_json;
        input_stream >> config_json;

        const auto base_dir = std::filesystem::absolute(input_path).parent_path();

        const auto intrinsics_path =
            resolve_path(base_dir, config_json.at("planar_intrinsics_config").get<std::string>());
        const auto planar_cfg = calib::planar::load_calibration_config(intrinsics_path);

        calib::pipeline::JsonPlanarDatasetLoader loader;
        for (const auto& entry : config_json.at("planar_detections")) {
            const std::string sensor_id = entry.at("sensor_id").get<std::string>();
            const auto features_path = resolve_path(base_dir, entry.at("path").get<std::string>());
            loader.add_entry(features_path, sensor_id);
        }

        calib::pipeline::PipelineContext context;
        context.set_intrinsics_config(planar_cfg);

        if (config_json.contains("stereo")) {
            auto stereo_cfg =
                config_json.at("stereo").get<calib::pipeline::StereoCalibrationConfig>();
            context.set_stereo_config(std::move(stereo_cfg));
        }

        calib::pipeline::CalibrationPipeline pipeline;
        if (verbose) {
            pipeline.add_decorator(std::make_shared<calib::pipeline::LoggingDecorator>(std::cerr));
        }
        pipeline.add_stage(std::make_unique<calib::pipeline::IntrinsicStage>());
        if (config_json.contains("stereo")) {
            pipeline.add_stage(std::make_unique<calib::pipeline::StereoCalibrationStage>());
        }

        const auto report = pipeline.execute(loader, context);

        nlohmann::json summary_json;
        summary_json["success"] = report.success;
        nlohmann::json stages_json = nlohmann::json::array();
        for (const auto& stage : report.stages) {
            nlohmann::json stage_json = stage.summary;
            stage_json["name"] = stage.name;
            stage_json["success"] = stage.success;
            stages_json.push_back(std::move(stage_json));
        }
        summary_json["stages"] = std::move(stages_json);

        context.artifacts["pipeline_summary"] = summary_json;

        // Optional: multi-camera extrinsics calibration using intrinsics result
        if (config_json.contains("multicam")) {
            std::vector<calib::pipeline::MultiCameraRigConfig> rigs;
            const auto& mc = config_json.at("multicam");
            if (mc.is_array()) {
                rigs = mc.get<std::vector<calib::pipeline::MultiCameraRigConfig>>();
            } else if (mc.is_object()) {
                rigs.push_back(mc.get<calib::pipeline::MultiCameraRigConfig>());
            }

            // Build detections and intrinsics lookup by sensor id from context
            std::unordered_map<std::string, calib::planar::PlanarDetections> det_by_sensor;
            for (const auto& det : context.dataset.planar_cameras) {
                if (!det.sensor_id.empty()) det_by_sensor.emplace(det.sensor_id, det);
            }
            std::unordered_map<std::string, calib::planar::CalibrationRunResult> intr_by_sensor(
                context.intrinsic_results.begin(), context.intrinsic_results.end());

            calib::pipeline::MultiCameraCalibrationFacade mc_facade;
            nlohmann::json multicam_artifacts;
            for (const auto& rig : rigs) {
                auto run = mc_facade.calibrate(rig, det_by_sensor, intr_by_sensor);

                nlohmann::json rig_json;
                rig_json["success"] = run.success;
                rig_json["requested_views"] = run.requested_views;
                rig_json["used_views"] = run.used_views;
                rig_json["sensors"] = run.sensors;

                nlohmann::json init_json;
                nlohmann::json cams_json = nlohmann::json::array();
                for (const auto& pose : run.initial_guess.c_se3_r) {
                    cams_json.push_back(pose);
                }
                nlohmann::json targets_json = nlohmann::json::array();
                for (const auto& pose : run.initial_guess.r_se3_t) {
                    targets_json.push_back(pose);
                }
                init_json["c_se3_r"] = std::move(cams_json);
                init_json["r_se3_t"] = std::move(targets_json);
                rig_json["initial_guess"] = std::move(init_json);
                rig_json["optimization"] = run.optimization;

                multicam_artifacts[rig.rig_id] = std::move(rig_json);
            }
            context.artifacts["multicam"] = std::move(multicam_artifacts);
        }

        std::ofstream output_stream(output_path);
        if (!output_stream) {
            throw std::runtime_error("Failed to open output file: " + output_path);
        }
        output_stream << context.artifacts.dump(2) << std::endl;

        std::cout << "Calibration pipeline completed. Artifacts written to " << output_path
                  << std::endl;
        return report.success ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "Calibration pipeline failed: " << ex.what() << std::endl;
        return 1;
    }
}
