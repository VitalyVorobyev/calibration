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

using calib::from_json;
using calib::to_json;

using namespace calib;

struct InputData final {
    PlanarView correspondences;
    std::optional<RansacOptions> ransac;
    bool optimize{true};
    HomographyOptions options;
};

struct OutputData final {
    bool success{false};
    int correspondence_count{0};
    HomographyResult estimated;
    std::optional<OptimizeHomographyResult> optimized;
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
    const InputData input_data = input_json;
    const bool run_refine = !disable_refine && input_data.optimize;

    auto initial = estimate_homography(input_data.correspondences, input_data.ransac);
    if (!initial.success) {
        std::cerr << "Failed to estimate homography";
        return 1;
    }

    OutputData output_data;
    output_data.success = true;
    output_data.correspondence_count = input_data.correspondences.size();
    output_data.estimated = std::move(initial);

    if (run_refine) {
        auto refined =
            optimize_homography(input_data.correspondences, initial.hmtx, input_data.options);
        output_data.optimized = std::move(refined);
    }

    nlohmann::json output_json = output_data;
    if (!output_path.empty()) {
        std::ofstream out_stream(output_path);
        if (!out_stream) {
            std::cerr << "Failed to open output file: " << output_path << "\n";
            return 1;
        }
        out_stream << output_json.dump(pretty ? 2 : -1) << '\n';
    } else {
        std::cout << output_json.dump(pretty ? 2 : -1) << '\n';
    }
}
