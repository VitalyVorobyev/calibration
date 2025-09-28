#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "calib/io/serialization.h"
#include "calib/pipeline/extrinsics.h"
#include "calib/pipeline/handeye.h"
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

[[nodiscard]] auto load_json_file(const std::filesystem::path& path) -> nlohmann::json {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }
    nlohmann::json json_data;
    stream >> json_data;
    return json_data;
}

[[nodiscard]] auto as_handeye_config(const nlohmann::json& node) -> calib::pipeline::HandEyePipelineConfig {
    calib::pipeline::HandEyePipelineConfig cfg;
    if (node.is_null()) {
        return cfg;
    }
    if (node.is_array()) {
        cfg.rigs = node.get<std::vector<calib::pipeline::HandEyeRigConfig>>();
    } else if (node.is_object() && node.contains("rigs")) {
        cfg = node.get<calib::pipeline::HandEyePipelineConfig>();
    } else {
        cfg.rigs.push_back(node.get<calib::pipeline::HandEyeRigConfig>());
    }
    return cfg;
}

[[nodiscard]] auto as_bundle_config(const nlohmann::json& node) -> calib::pipeline::BundlePipelineConfig {
    calib::pipeline::BundlePipelineConfig cfg;
    if (node.is_null()) {
        return cfg;
    }
    if (node.is_array()) {
        cfg.rigs = node.get<std::vector<calib::pipeline::BundleRigConfig>>();
    } else if (node.is_object() && node.contains("rigs")) {
        cfg = node.get<calib::pipeline::BundlePipelineConfig>();
    } else {
        cfg.rigs.push_back(node.get<calib::pipeline::BundleRigConfig>());
    }
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Planar intrinsics + hand-eye + bundle adjustment calibration pipeline"};

    std::string input_path;
    std::string output_path = "bundle_artifacts.json";
    bool verbose = false;

    app.add_option("--input", input_path, "Pipeline input configuration JSON")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--output", output_path, "Path to write calibration artifacts JSON");
    app.add_flag("-v,--verbose", verbose, "Print pipeline stage transitions");

    CLI11_PARSE(app, argc, argv);

    try {
        const auto config_json = load_json_file(input_path);
        const auto base_dir = std::filesystem::absolute(input_path).parent_path();

        const auto intrinsics_cfg_path = resolve_path(
            base_dir, config_json.at("planar_intrinsics_config").get<std::string>());
        const auto planar_cfg = calib::planar::load_calibration_config(intrinsics_cfg_path);

        calib::pipeline::JsonPlanarDatasetLoader loader;
        for (const auto& entry : config_json.at("planar_detections")) {
            const std::string sensor_id = entry.at("sensor_id").get<std::string>();
            const auto features_path = resolve_path(base_dir, entry.at("path").get<std::string>());
            loader.add_entry(features_path, sensor_id);
        }

        calib::pipeline::PipelineContext context;
        context.set_intrinsics_config(planar_cfg);

        if (config_json.contains("stereo")) {
            auto stereo_cfg = config_json.at("stereo").get<calib::pipeline::StereoCalibrationConfig>();
            context.set_stereo_config(std::move(stereo_cfg));
        }

        if (config_json.contains("hand_eye")) {
            auto he_cfg = as_handeye_config(config_json.at("hand_eye"));
            if (!he_cfg.rigs.empty()) {
                context.set_handeye_config(std::move(he_cfg));
            }
        }

        if (config_json.contains("bundle")) {
            auto bundle_cfg = as_bundle_config(config_json.at("bundle"));
            if (!bundle_cfg.rigs.empty()) {
                context.set_bundle_config(std::move(bundle_cfg));
            }
        }

        calib::pipeline::CalibrationPipeline pipeline;
        if (verbose) {
            pipeline.add_decorator(std::make_shared<calib::pipeline::LoggingDecorator>(std::cerr));
        }

        pipeline.add_stage(std::make_unique<calib::pipeline::IntrinsicStage>());
        if (context.has_stereo_config()) {
            pipeline.add_stage(std::make_unique<calib::pipeline::StereoCalibrationStage>());
        }
        if (context.has_handeye_config()) {
            pipeline.add_stage(std::make_unique<calib::pipeline::HandEyeCalibrationStage>());
        }
        if (context.has_bundle_config()) {
            pipeline.add_stage(std::make_unique<calib::pipeline::BundleAdjustmentStage>());
        }

        auto report = pipeline.execute(loader, context);

        nlohmann::json pipeline_summary;
        pipeline_summary["success"] = report.success;
        nlohmann::json stages_json = nlohmann::json::array();
        for (const auto& stage : report.stages) {
            nlohmann::json stage_json = stage.summary;
            stage_json["name"] = stage.name;
            stage_json["success"] = stage.success;
            stages_json.push_back(std::move(stage_json));
        }
        pipeline_summary["stages"] = std::move(stages_json);
        context.artifacts["pipeline_summary"] = pipeline_summary;

        std::ofstream output_stream(output_path);
        if (!output_stream) {
            throw std::runtime_error("Failed to open output file: " + output_path);
        }
        output_stream << context.artifacts.dump(2) << '\n';

        std::cout << "Calibration pipeline completed. Artifacts written to " << output_path
                  << std::endl;
        return report.success ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "Calibration pipeline failed: " << ex.what() << std::endl;
        return 1;
    }
}

