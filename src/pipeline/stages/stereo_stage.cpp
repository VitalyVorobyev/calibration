#include "calib/pipeline/stages.h"
#include "calib/pipeline/facades/extrinsics.h"
#include "calib/pipeline/detail/planar_utils.h"

namespace calib::pipeline {

namespace {

using detail::build_sensor_index;

auto build_detection_lookup(const std::vector<PlanarDetections>& detections) -> std::unordered_map<std::string, const PlanarDetections*> {
    std::unordered_map<std::string, const PlanarDetections*> lookup;
    for (const auto& det : detections) {
        if (!det.sensor_id.empty()) {
            lookup.emplace(det.sensor_id, &det);
        }
    }
    return lookup;
}

nlohmann::json build_missing_list(const std::vector<std::string>& ids) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& id : ids) {
        arr.push_back(id);
    }
    return arr;
}

}  // namespace

// TODO: 1. refactor, 2. support multi-camera rigs
auto StereoCalibrationStage::run(PipelineContext& context) -> PipelineStageResult {
    PipelineStageResult result;
    result.name = name();
    const std::size_t calibrated_cameras = context.intrinsic_results.size();
    result.summary["input_cameras"] = calibrated_cameras;

    if (!context.has_stereo_config()) {
        result.summary["status"] = "missing_config";
        result.success = false;
        return result;
    }
    if (calibrated_cameras < 2) {
        result.summary["status"] = "waiting_for_multiple_intrinsic_results";
        result.success = false;
        return result;
    }

    const auto& stereo_cfg = context.stereo_config();
    result.summary["requested_pairs"] = stereo_cfg.pairs.size();
    if (stereo_cfg.pairs.empty()) {
        result.summary["status"] = "no_pairs_configured";
        result.success = false;
        return result;
    }

    const auto detections_by_sensor = build_detection_lookup(context.dataset.planar_cameras);

    if (!context.artifacts.is_object()) {
        context.artifacts = nlohmann::json::object();
    }
    auto& stereo_artifacts = context.artifacts["stereo"];
    if (!stereo_artifacts.is_object()) {
        stereo_artifacts = nlohmann::json::object();
    }
    stereo_artifacts["pairs"] = nlohmann::json::object();
    context.stereo_results.clear();

    StereoCalibrationFacade facade;
    nlohmann::json pairs_summary = nlohmann::json::array();

    bool all_success = true;
    bool any_success = false;

    for (const auto& pair_cfg : stereo_cfg.pairs) {
        nlohmann::json pair_json;
        pair_json["pair_id"] = pair_cfg.pair_id;
        pair_json["reference_sensor"] = pair_cfg.reference_sensor;
        pair_json["target_sensor"] = pair_cfg.target_sensor;
        pair_json["requested_views"] = pair_cfg.views.size();

        const auto ref_intr_it = context.intrinsic_results.find(pair_cfg.reference_sensor);
        const auto tgt_intr_it = context.intrinsic_results.find(pair_cfg.target_sensor);

        if (ref_intr_it == context.intrinsic_results.end() ||
            tgt_intr_it == context.intrinsic_results.end()) {
            std::vector<std::string> missing;
            if (ref_intr_it == context.intrinsic_results.end())
                missing.push_back(pair_cfg.reference_sensor);
            if (tgt_intr_it == context.intrinsic_results.end())
                missing.push_back(pair_cfg.target_sensor);
            pair_json["status"] = "missing_intrinsics";
            pair_json["missing"] = build_missing_list(missing);
            pair_json["success"] = false;
            all_success = false;
            pairs_summary.push_back(std::move(pair_json));
            continue;
        }

        const auto ref_det_it = detections_by_sensor.find(pair_cfg.reference_sensor);
        const auto tgt_det_it = detections_by_sensor.find(pair_cfg.target_sensor);
        if (ref_det_it == detections_by_sensor.end() || tgt_det_it == detections_by_sensor.end()) {
            std::vector<std::string> missing;
            if (ref_det_it == detections_by_sensor.end())
                missing.push_back(pair_cfg.reference_sensor);
            if (tgt_det_it == detections_by_sensor.end()) missing.push_back(pair_cfg.target_sensor);
            pair_json["status"] = "missing_detections";
            pair_json["missing"] = build_missing_list(missing);
            pair_json["success"] = false;
            all_success = false;
            pairs_summary.push_back(std::move(pair_json));
            continue;
        }

        try {
            auto pair_result = facade.calibrate(pair_cfg, *ref_det_it->second, *tgt_det_it->second,
                                                ref_intr_it->second, tgt_intr_it->second);

            nlohmann::json views_json = nlohmann::json::array();
            for (const auto& view : pair_result.view_summaries) {
                views_json.push_back(view);
            }

            pair_json["views"] = views_json;
            pair_json["used_views"] = pair_result.used_views;
            pair_json["success"] = pair_result.success;
            pair_json["status"] = pair_result.success ? "ok" : "failed";
            pair_json["final_cost"] = pair_result.optimization.final_cost;

            if (pair_result.success) {
                any_success = true;
                context.stereo_results[pair_cfg.pair_id] = pair_result.optimization;
            } else {
                all_success = false;
            }

            nlohmann::json initial_guess_json;
            nlohmann::json cams_json = nlohmann::json::array();
            std::copy(pair_result.initial_guess.c_se3_r.begin(),
                      pair_result.initial_guess.c_se3_r.end(), std::back_inserter(cams_json));
            nlohmann::json targets_json = nlohmann::json::array();
            std::copy(pair_result.initial_guess.r_se3_t.begin(),
                      pair_result.initial_guess.r_se3_t.end(), std::back_inserter(targets_json));
            initial_guess_json["c_se3_r"] = std::move(cams_json);
            initial_guess_json["r_se3_t"] = std::move(targets_json);

            nlohmann::json artifact;
            artifact["initial_guess"] = std::move(initial_guess_json);
            artifact["views"] = pair_json["views"];
            artifact["optimization"] = pair_result.optimization;
            artifact["final_cost"] = pair_result.optimization.final_cost;
            stereo_artifacts["pairs"][pair_cfg.pair_id] = std::move(artifact);
        } catch (const std::exception& ex) {
            pair_json["status"] = "calibration_error";
            pair_json["error"] = ex.what();
            pair_json["success"] = false;
            all_success = false;
        }

        pairs_summary.push_back(std::move(pair_json));
    }

    result.summary["pairs"] = std::move(pairs_summary);
    if (any_success && all_success) {
        result.summary["status"] = "ok";
        result.success = true;
    } else if (any_success) {
        result.summary["status"] = "partial_success";
        result.success = false;
    } else {
        result.summary["status"] = "failed";
        result.success = false;
    }

    return result;
}

}  // namespace calib::pipeline
