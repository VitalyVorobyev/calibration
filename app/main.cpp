#include <CLI/CLI.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

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
            // Read observations for each camera from input JSON
            std::ifstream in_stream(app_config.input_path);
            if (!in_stream) {
                throw std::runtime_error("Failed to open input: " + app_config.input_path);
            }
            nlohmann::json in_json;
            in_stream >> in_json;
            nlohmann::json cam_results = nlohmann::json::array();

            for (const auto& cam_views_j : in_json.at("cameras")) {
                std::vector<PlanarView> views;
                std::vector<Observation<double>> all_obs;

                for (const auto& view_j : cam_views_j) {
                    PlanarView view;
                    for (const auto& obs_j : view_j) {
                        auto obs = obs_j.get<Observation<double>>();
                        PlanarObservation po;
                        po.object_xy = Eigen::Vector2d(obs.x, obs.y);
                        po.image_uv = Eigen::Vector2d(obs.u, obs.v);
                        view.push_back(po);
                        all_obs.push_back(obs);
                    }
                    views.push_back(view);
                }

                auto init_cam_opt = estimate_intrinsics_linear_iterative(all_obs, 2);
                if (!init_cam_opt) {
                    throw std::runtime_error("Failed to initialise intrinsics");
                }

                std::vector<Eigen::Isometry3d> init_poses;
                for (const auto& view : views) {
                    std::vector<Eigen::Vector2d> obj_xy;
                    std::vector<Eigen::Vector2d> img_uv;
                    obj_xy.reserve(view.size());
                    img_uv.reserve(view.size());
                    for (const auto& obs : view) {
                        obj_xy.push_back(obs.object_xy);
                        img_uv.push_back(obs.image_uv);
                    }
                    init_poses.push_back(estimate_planar_pose_dlt(obj_xy, img_uv, init_cam_opt->kmtx));
                }

                IntrinsicsOptions opts;
                auto cam_res = optimize_intrinsics(views, *init_cam_opt, init_poses, opts);
                cam_results.push_back(cam_res);
            }

            result = nlohmann::json{{"cameras", cam_results}};
        #if 0
        } else if (task == "extrinsics") {
            ExtrinsicsInput in = cfg.at("input").get<ExtrinsicsInput>();
            auto guess = estimate_extrinsic_dlt(in.views, in.cameras);
            auto r = optimize_extrinsic_poses(in.views, in.cameras, guess.camera_poses, guess.target_poses);
            result = r;
        // } else if (task == "handeye") {
        //     HandEyeInput in = cfg.at("input").get<HandEyeInput>();
        //     auto r = refine_hand_eye_reprojection(in.base_se3_gripper, in.views, in.intrinsics, in.init_gripper_se3_ref, in.options);
        //     result = r;
        #endif
        } else if (task == "bundle") {
            BundleInput in = cfg.at("input").get<BundleInput>();
            auto r = optimize_bundle(in.observations, in.initial_cameras, in.init_g_se3_c, in.init_b_se3_t, in.options);
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
