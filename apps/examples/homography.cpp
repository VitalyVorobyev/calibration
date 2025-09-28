// Homography estimation & refinement example with JSON I/O

#include "calib/estimation/optim/homography.h"

#include <CLI/CLI.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <utility>

#include "calib/io/serialization.h"

using namespace calib;

struct InputData final {
    PlanarView correspondences;
    std::optional<RansacOptions> ransac;
    bool optimize{true};
    HomographyOptions options;
};

int main(int argc, char** argv) {
    CLI::App app{"Homography estimation and refinement example"};

    std::string input_path;
    std::string output_path;
    bool pretty = false;
    bool disable_refine = false;

    app.add_option("--input", input_path, "Input JSON with correspondences")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-o,--output", output_path, "Optional output JSON file (default stdout)");
    app.add_flag("--pretty", pretty, "Pretty-print JSON output");
    app.add_flag("--no-refine", disable_refine, "Skip non-linear refinement step");

    CLI11_PARSE(app, argc, argv);

    nlohmann::json input_json;
    {
        std::ifstream input_stream(input_path);
        if (!input_stream) {
            std::cerr << "Failed to open input file: " << input_path << "\n";
            return 1;
        }
        input_stream >> input_json;
    }

    PlanarView view = input_json.at("correspondences").get<PlanarView>();

    std::optional<RansacOptions> ransac_opts;
    if (input_json.contains("ransac")) {
        ransac_opts = input_json.at("ransac").get<RansacOptions>();
    }

    bool run_refine = !disable_refine;
    run_refine = input_json.value("optimize", run_refine) && !disable_refine;

    HomographyOptions optim_opts;
    if (input_json.contains("options")) {
        optim_opts = input_json.at("options").get<HomographyOptions>();
    }

    auto initial = estimate_homography(view, ransac_opts);
    if (!initial.success) {
        std::cerr << "Failed to estimate homography";
        return 1;
    }

    nlohmann::json output;
    output["success"] = true;
    output["correspondence_count"] = view.size();

    nlohmann::json initial_json;
    initial_json["homography"] = initial.hmtx;
    initial_json["symmetric_rms_px"] = initial.symmetric_rms_px;
    if (!initial.inliers.empty()) {
        initial_json["inliers"] = initial.inliers;
        initial_json["inlier_count"] = initial.inliers.size();
    }
    output["initial"] = std::move(initial_json);

    if (run_refine) {
        auto refined = optimize_homography(view, initial.hmtx, optim_opts);
        nlohmann::json refined_json;
        refined_json["success"] = refined.success;
        refined_json["homography"] = refined.homography;
        refined_json["final_cost"] = refined.final_cost;
        refined_json["report"] = refined.report;
        if (refined.covariance.size() != 0) {
            refined_json["covariance"] = refined.covariance;
        }
        output["optimized"] = std::move(refined_json);
    } else {
        output["optimized"] = nullptr;
    }

    auto dump_json = [&](std::ostream& os) {
        if (pretty) {
            os << output.dump(2) << '\n';
        } else {
            os << output.dump() << '\n';
        }
    };

    if (!output_path.empty()) {
        std::ofstream out_stream(output_path);
        if (!out_stream) {
            std::cerr << "Failed to open output file: " << output_path << "\n";
            return 1;
        }
        dump_json(out_stream);
    } else {
        dump_json(std::cout);
    }

    return 0;
}
