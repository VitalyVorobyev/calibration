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

namespace {

auto matrix_to_json(const Eigen::Matrix3d& mat) -> nlohmann::json {
    nlohmann::json out = nlohmann::json::array();
    for (int r = 0; r < mat.rows(); ++r) {
        nlohmann::json row = nlohmann::json::array();
        for (int c = 0; c < mat.cols(); ++c) {
            row.push_back(mat(r, c));
        }
        out.push_back(std::move(row));
    }
    return out;
}

auto covariance_to_json(const Eigen::MatrixXd& cov) -> nlohmann::json {
    if (cov.size() == 0) {
        return nlohmann::json();
    }
    nlohmann::json out = nlohmann::json::array();
    for (Eigen::Index r = 0; r < cov.rows(); ++r) {
        nlohmann::json row = nlohmann::json::array();
        for (Eigen::Index c = 0; c < cov.cols(); ++c) {
            row.push_back(cov(r, c));
        }
        out.push_back(std::move(row));
    }
    return out;
}

auto parse_correspondences(const nlohmann::json& node) -> PlanarView {
    if (!node.is_array() || node.size() < 4) {
        throw std::runtime_error("'correspondences' must be an array with at least 4 entries");
    }

    PlanarView view;
    view.reserve(node.size());
    for (const auto& entry : node) {
        PlanarObservation obs;

        if (entry.contains("object") && entry.contains("image")) {
            const auto& obj = entry.at("object");
            const auto& img = entry.at("image");
            if (!obj.is_array() || obj.size() != 2 || !img.is_array() || img.size() != 2) {
                throw std::runtime_error("object/image must be 2-element arrays");
            }
            obs.object_xy = Eigen::Vector2d(obj[0].get<double>(), obj[1].get<double>());
            obs.image_uv = Eigen::Vector2d(img[0].get<double>(), img[1].get<double>());
        } else if (entry.is_array() && entry.size() == 4) {
            obs.object_xy = Eigen::Vector2d(entry[0].get<double>(), entry[1].get<double>());
            obs.image_uv = Eigen::Vector2d(entry[2].get<double>(), entry[3].get<double>());
        } else {
            throw std::runtime_error(
                "Each correspondence must be object/image arrays or [x,y,u,v] tuple");
        }

        view.push_back(std::move(obs));
    }
    return view;
}

auto parse_ransac(const nlohmann::json& node) -> std::optional<RansacOptions> {
    if (!node.is_object()) {
        return std::nullopt;
    }

    RansacOptions opts;
    if (node.contains("max_iters")) {
        opts.max_iters = node.at("max_iters").get<int>();
    }
    if (node.contains("thresh")) {
        opts.thresh = node.at("thresh").get<double>();
    }
    if (node.contains("min_inliers")) {
        opts.min_inliers = node.at("min_inliers").get<int>();
    }
    if (node.contains("confidence")) {
        opts.confidence = node.at("confidence").get<double>();
    }
    if (node.contains("seed")) {
        opts.seed = node.at("seed").get<uint64_t>();
    }
    if (node.contains("refit_on_inliers")) {
        opts.refit_on_inliers = node.at("refit_on_inliers").get<bool>();
    } else if (node.contains("refit")) {
        opts.refit_on_inliers = node.at("refit").get<bool>();
    }
    return opts;
}

void parse_homography_options(const nlohmann::json& node, HomographyOptions& opts) {
    if (!node.is_object()) {
        return;
    }
    if (node.contains("optimizer")) {
        const auto& opt = node.at("optimizer");
        if (opt.is_string()) {
            opts.optimizer = optimizer_type_from_string(opt.get<std::string>());
        } else if (opt.is_number_integer()) {
            opts.optimizer = static_cast<OptimizerType>(opt.get<int>());
        }
    }
    if (node.contains("huber_delta")) {
        opts.huber_delta = node.at("huber_delta").get<double>();
    }
    if (node.contains("epsilon")) {
        opts.epsilon = node.at("epsilon").get<double>();
    }
    if (node.contains("max_iterations")) {
        opts.max_iterations = node.at("max_iterations").get<int>();
    }
    if (node.contains("compute_covariance")) {
        opts.compute_covariance = node.at("compute_covariance").get<bool>();
    }
    if (node.contains("verbose")) {
        opts.verbose = node.at("verbose").get<bool>();
    }
}

}  // namespace

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

    PlanarView view = parse_correspondences(input_json.at("correspondences"));

    std::optional<RansacOptions> ransac_opts;
    if (input_json.contains("ransac")) {
        ransac_opts = parse_ransac(input_json.at("ransac"));
    }

    bool run_refine = !disable_refine;
    if (input_json.contains("optimize")) {
        run_refine = input_json.at("optimize").get<bool>() && !disable_refine;
    }

    HomographyOptions optim_opts;
    if (input_json.contains("options")) {
        parse_homography_options(input_json.at("options"), optim_opts);
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
    initial_json["homography"] = matrix_to_json(initial.hmtx);
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
        refined_json["homography"] = matrix_to_json(refined.homography);
        refined_json["final_cost"] = refined.final_cost;
        refined_json["report"] = refined.report;
        nlohmann::json cov_json = covariance_to_json(refined.covariance);
        if (!cov_json.is_null() && !cov_json.empty()) {
            refined_json["covariance"] = std::move(cov_json);
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
