#include "calib/pipeline/stages.h"

#include "calib/io/serialization.h"
#include "calib/pipeline/extrinsics.h"

// std
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

namespace calib::pipeline {

namespace {

[[nodiscard]] auto find_camera_config(const planar::PlanarCalibrationConfig& cfg,
                                      const std::string& camera_id) -> const planar::CameraConfig* {
    const auto it =
        std::find_if(cfg.cameras.begin(), cfg.cameras.end(),
                     [&](const planar::CameraConfig& cam) { return cam.camera_id == camera_id; });
    return it == cfg.cameras.end() ? nullptr : &(*it);
}

}  // namespace

auto IntrinsicStage::run(PipelineContext& context) -> PipelineStageResult {
    PipelineStageResult result;
    result.name = name();

    if (!context.has_intrinsics_config()) {
        result.summary["error"] = "No intrinsics configuration supplied.";
        result.success = false;
        return result;
    }

    if (context.dataset.planar_cameras.empty()) {
        result.summary["error"] = "Dataset does not contain planar camera captures.";
        result.success = false;
        return result;
    }

    const auto& cfg = context.intrinsics_config();
    planar::PlanarIntrinsicCalibrationFacade facade;

    bool success = true;
    nlohmann::json cameras_summary = nlohmann::json::array();

    for (const auto& detections : context.dataset.planar_cameras) {
        const std::string sensor_id = !detections.sensor_id.empty() ? detections.sensor_id : "cam0";
        const auto* cam_cfg = find_camera_config(cfg, sensor_id);
        if (cam_cfg == nullptr) {
            success = false;
            cameras_summary.push_back(
                {{"sensor_id", sensor_id}, {"status", "missing_camera_config"}});
            continue;
        }

        try {
            auto run_result = facade.calibrate(cfg, *cam_cfg, detections, detections.source_file);
            context.intrinsic_results[sensor_id] = run_result;

            nlohmann::json camera_entry = run_result.report;
            camera_entry["sensor_id"] = sensor_id;
            nlohmann::json tag_array = nlohmann::json::array();
            std::copy(detections.tags.begin(), detections.tags.end(),
                      std::back_inserter(tag_array));
            camera_entry["tags"] = std::move(tag_array);
            cameras_summary.push_back(std::move(camera_entry));
        } catch (const std::exception& ex) {
            success = false;
            cameras_summary.push_back(
                {{"sensor_id", sensor_id}, {"status", "calibration_failed"}, {"error", ex.what()}});
        }
    }

    bool has_synthetic = false;
    bool has_recorded = false;
    for (const auto& detections : context.dataset.planar_cameras) {
        if (detections.tags.count("synthetic")) {
            has_synthetic = true;
        }
        if (detections.tags.count("recorded")) {
            has_recorded = true;
        }
    }

    result.summary["cameras"] = std::move(cameras_summary);
    result.summary["gating"] = {{"synthetic", has_synthetic}, {"recorded", has_recorded}};
    result.success = success && !context.intrinsic_results.empty();
    return result;
}

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

    std::unordered_map<std::string, const planar::PlanarDetections*> detections_by_sensor;
    for (const auto& detections : context.dataset.planar_cameras) {
        if (!detections.sensor_id.empty()) {
            detections_by_sensor.emplace(detections.sensor_id, &detections);
        }
    }

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
            pair_json["status"] = "missing_intrinsics";
            nlohmann::json missing = nlohmann::json::array();
            if (ref_intr_it == context.intrinsic_results.end()) {
                missing.push_back(pair_cfg.reference_sensor);
            }
            if (tgt_intr_it == context.intrinsic_results.end()) {
                missing.push_back(pair_cfg.target_sensor);
            }
            pair_json["missing"] = std::move(missing);
            pair_json["success"] = false;
            all_success = false;
            pairs_summary.push_back(std::move(pair_json));
            continue;
        }

        const auto ref_det_it = detections_by_sensor.find(pair_cfg.reference_sensor);
        const auto tgt_det_it = detections_by_sensor.find(pair_cfg.target_sensor);
        if (ref_det_it == detections_by_sensor.end() || tgt_det_it == detections_by_sensor.end()) {
            pair_json["status"] = "missing_detections";
            nlohmann::json missing = nlohmann::json::array();
            if (ref_det_it == detections_by_sensor.end()) {
                missing.push_back(pair_cfg.reference_sensor);
            }
            if (tgt_det_it == detections_by_sensor.end()) {
                missing.push_back(pair_cfg.target_sensor);
            }
            pair_json["missing"] = std::move(missing);
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
                nlohmann::json view_json;
                to_json(view_json, view);
                views_json.push_back(std::move(view_json));
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
            for (const auto& pose : pair_result.initial_guess.c_se3_r) {
                cams_json.push_back(calib::affine_to_json(pose));
            }
            nlohmann::json targets_json = nlohmann::json::array();
            for (const auto& pose : pair_result.initial_guess.r_se3_t) {
                targets_json.push_back(calib::affine_to_json(pose));
            }
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

auto HandEyeCalibrationStage::run(PipelineContext& context) -> PipelineStageResult {
    PipelineStageResult result;
    result.name = name();

    if (context.intrinsic_results.empty()) {
        result.summary["status"] = "waiting_for_intrinsic_stage";
        result.success = false;
        return result;
    }

    // Placeholder for future hand-eye calibration implementation.
    result.summary["status"] = "not_implemented";
    result.success = true;
    return result;
}

}  // namespace calib::pipeline
