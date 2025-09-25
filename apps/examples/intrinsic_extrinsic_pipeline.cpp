#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "calib/pipeline/extrinsics.h"
#include "calib/pipeline/loaders.h"
#include "calib/pipeline/pipeline.h"
#include "calib/pipeline/stages.h"

namespace {

[[nodiscard]] auto resolve_path(const std::filesystem::path& base,
                                const std::string& candidate) -> std::filesystem::path {
    const std::filesystem::path path(candidate);
    if (path.is_absolute()) {
        return path;
    }
    return base / path;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Planar intrinsics and stereo extrinsics calibration example"};

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

        if (!config_json.contains("stereo")) {
            throw std::runtime_error("Input configuration must provide a 'stereo' section.");
        }
        auto stereo_cfg = config_json.at("stereo").get<calib::pipeline::StereoCalibrationConfig>();
        context.set_stereo_config(std::move(stereo_cfg));

        calib::pipeline::CalibrationPipeline pipeline;
        if (verbose) {
            pipeline.add_decorator(std::make_shared<calib::pipeline::LoggingDecorator>(std::cerr));
        }
        pipeline.add_stage(std::make_unique<calib::pipeline::IntrinsicStage>());
        pipeline.add_stage(std::make_unique<calib::pipeline::StereoCalibrationStage>());

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

