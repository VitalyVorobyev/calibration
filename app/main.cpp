#include <CLI/CLI.hpp>

#include <fstream>
#include <iostream>

#include "calib/serialization.h"
#include "config.h"

using namespace calib;

int main(int argc, char** argv) {
    CLI::App app{"Calibration app"};

    std::string config_path;
    std::string output_override;
    std::string task_override;

    app.add_option("-c,--config", config_path, "Calibration config file")->required();
    app.add_option("-t,--task", task_override, "Override task from config");
    app.add_option("-o,--output", output_override, "Override output file");

    CLI11_PARSE(app, argc, argv);

    std::ifstream cfg_stream(config_path);
    if (!cfg_stream) {
        std::cerr << "Failed to open config: " << config_path << std::endl;
        return 1;
    }
    nlohmann::json cfg; cfg_stream >> cfg;
    AppConfig app_config = cfg;

    if (!task_override.empty()) app_config.task = task_override;
    if (!output_override.empty()) app_config.output = output_override;

    if (app_config.task.empty()) {
        std::cerr << "Task not specified in config" << std::endl;
        return 1;
    }

    std::string task = app_config.task;
    nlohmann::json result;

    try {
        if (task == "intrinsics") {
            result = nlohmann::json{{"error", "intrinsics task not supported"}};
        } else if (task == "extrinsics") {
            ExtrinsicsInput in = cfg.at("input").get<ExtrinsicsInput>();
            auto guess = make_initial_extrinsic_guess(in.views, in.cameras);
            auto r = optimize_extrinsic_poses(in.views, in.cameras, guess.camera_poses, guess.target_poses);
            result = r;
        // } else if (task == "handeye") {
        //     HandEyeInput in = cfg.at("input").get<HandEyeInput>();
        //     auto r = refine_hand_eye_reprojection(in.base_T_gripper, in.views, in.intrinsics, in.init_gripper_T_ref, in.options);
        //     result = r;
        } else if (task == "bundle") {
            BundleInput in = cfg.at("input").get<BundleInput>();
            auto r = optimize_bundle(in.observations, in.initial_cameras, in.init_g_T_c, in.init_b_T_t, in.options);
            result = r;
        } else {
            std::cerr << "Unknown task: " << task << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Calibration failed: " << e.what() << std::endl;
        return 1;
    }

    std::string out_path = app_config.output;
    if (!out_path.empty()) {
        std::ofstream out(out_path);
        out << result.dump(2);
    }
    std::cout << result.dump(2) << std::endl;
}
