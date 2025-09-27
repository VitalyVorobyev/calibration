#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "calib/estimation/linescan.h"
#include "calib/io/serialization.h"
#include "calib/pipeline/linescan.h"

int main(int argc, char** argv) {
    CLI::App app{"Line-scan laser plane calibration (linear)"};
    std::string input_path;
    std::string output_path = "linescan_artifacts.json";
    app.add_option("--input", input_path, "Input JSON (camera, views)")->required()->check(CLI::ExistingFile);
    app.add_option("--output", output_path, "Output JSON path");
    CLI11_PARSE(app, argc, argv);

    try {
        std::ifstream in(input_path);
        if (!in) throw std::runtime_error("Cannot open input file");
        nlohmann::json j;
        in >> j;

        calib::PinholeCamera<calib::BrownConradyd> camera;
        j.at("camera").get_to(camera);

        std::vector<calib::LineScanView> views;
        for (const auto& vj : j.at("views")) {
            calib::LineScanView v;
            vj.at("target_view").get_to(v.target_view);
            v.laser_uv.clear();
            for (const auto& arr : vj.at("laser_uv")) {
                if (!arr.is_array() || arr.size() != 2) {
                    throw std::runtime_error("laser_uv entry must be [u,v]");
                }
                v.laser_uv.emplace_back(arr[0].get<double>(), arr[1].get<double>());
            }
            views.push_back(std::move(v));
        }

        calib::pipeline::LinescanCalibrationFacade facade;
        auto run = facade.calibrate(camera, views);

        nlohmann::json out;
        out["success"] = run.success;
        out["used_views"] = run.used_views;
        out["plane"] = {{"n", {run.result.plane[0], run.result.plane[1], run.result.plane[2]}},
                         {"d", run.result.plane[3]}};
        out["rms_error"] = run.result.rms_error;
        out["homography"] = calib::eigen_matrix_to_json(run.result.homography);

        std::ofstream out_stream(output_path);
        if (!out_stream) throw std::runtime_error("Cannot open output file");
        out_stream << out.dump(2) << std::endl;
        std::cout << "Linescan calibration artifacts written to " << output_path << std::endl;
        return run.success ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "Linescan calibration failed: " << ex.what() << std::endl;
        return 1;
    }
}
