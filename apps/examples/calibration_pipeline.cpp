#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "calib/pipeline/dataset.h"
#include "calib/pipeline/loaders.h"
#include "calib/pipeline/stages.h"
#include "calib/pipeline/facades/intrinsics.h"

namespace {

[[nodiscard]] auto split_sensor_entry(const std::string& arg)
    -> std::pair<std::optional<std::string>, std::filesystem::path> {
    const auto pos = arg.find('=');
    if (pos == std::string::npos) {
        return {std::nullopt, std::filesystem::path(arg)};
    }
    auto sensor = arg.substr(0, pos);
    auto path = arg.substr(pos + 1);
    return {sensor.empty() ? std::optional<std::string>{} : std::optional<std::string>{sensor},
            std::filesystem::path(path)};
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"End-to-end calibration pipeline (intrinsics → stereo → hand-eye)"};

    std::string config_path;
    std::vector<std::string> feature_args;
    bool verbose = false;

    app.add_option("--config", config_path, "Planar calibration configuration")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--features", feature_args,
                   "Feature dataset files. Accepts path or sensor_id=path syntax.")
        ->required();
    app.add_flag("-v,--verbose", verbose, "Print pipeline stage transitions");

    CLI11_PARSE(app, argc, argv);

    try {
        const auto config = calib::pipeline::load_calibration_config(config_path);
        if (!config.has_value()) {
            throw std::runtime_error("Failed to load calibration config from " + config_path);
        }

        calib::pipeline::JsonPlanarDatasetLoader loader;
        for (const auto& entry : feature_args) {
            auto [sensor_id, path] = split_sensor_entry(entry);
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("Feature file not found: " + path.string());
            }
            loader.add_entry(path, sensor_id);
        }

        calib::pipeline::PipelineContext context;
        context.set_intrinsics_config(config.value());

        calib::pipeline::CalibrationPipeline pipeline;
        if (verbose) {
            pipeline.add_decorator(std::make_shared<calib::pipeline::LoggingDecorator>(std::cerr));
        }
        pipeline.add_stage(std::make_unique<calib::pipeline::IntrinsicStage>());
        pipeline.add_stage(std::make_unique<calib::pipeline::StereoCalibrationStage>());
        pipeline.add_stage(std::make_unique<calib::pipeline::HandEyeCalibrationStage>());

        const auto report = pipeline.execute(loader, context);

        nlohmann::json json_report;
        json_report["success"] = report.success;
        nlohmann::json stages_json = nlohmann::json::array();
        for (const auto& stage : report.stages) {
            nlohmann::json stage_json = stage.summary;
            stage_json["name"] = stage.name;
            stage_json["success"] = stage.success;
            stages_json.push_back(std::move(stage_json));
        }
        json_report["stages"] = std::move(stages_json);

        std::cout << json_report.dump(2) << std::endl;
        return report.success ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "Pipeline execution failed: " << ex.what() << std::endl;
        return 1;
    }
}
