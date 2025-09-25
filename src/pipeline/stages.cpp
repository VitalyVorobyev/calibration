#include "calib/pipeline/stages.h"

// std
#include <algorithm>
#include <iterator>
#include <stdexcept>

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
    if (calibrated_cameras < 2) {
        result.summary["status"] = "waiting_for_multiple_intrinsic_results";
        result.success = false;
        return result;
    }

    // Placeholder for future stereo calibration implementation.
    result.summary["status"] = "not_implemented";
    result.success = true;
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
