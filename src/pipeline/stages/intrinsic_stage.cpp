#include "calib/pipeline/detail/planar_utils.h"
#include "calib/pipeline/facades/intrinsics.h"
#include "calib/pipeline/reports/intrinsics.h"
#include "calib/pipeline/stages.h"

namespace calib::pipeline {

namespace {

using detail::find_camera_config;

struct SensorCalibrationResult {
    bool success{false};
    nlohmann::json summary;
};

SensorCalibrationResult calibrate_sensor(const PlanarIntrinsicCalibrationFacade& facade,
                                         const IntrinsicCalibrationConfig& cfg,
                                         const PlanarDetections& detections,
                                         PipelineContext& context) {
    SensorCalibrationResult result;
    const std::string sensor_id = !detections.sensor_id.empty() ? detections.sensor_id : "cam0";
    const auto* cam_cfg = find_camera_config(cfg, sensor_id);
    if (cam_cfg == nullptr) {
        result.summary = nlohmann::json{
            {"sensor_id", sensor_id},
            {"status", "missing_camera_config"},
        };
        return result;
    }

    try {
        auto run = facade.calibrate(cfg, *cam_cfg, detections);
        context.intrinsic_results[sensor_id] = run;

        const auto report = build_planar_intrinsics_report(cfg, *cam_cfg, detections, run);
        nlohmann::json entry;
        to_json(entry, report);
        entry["sensor_id"] = sensor_id;
        nlohmann::json tags = nlohmann::json::array();
        std::copy(detections.tags.begin(), detections.tags.end(), std::back_inserter(tags));
        entry["tags"] = std::move(tags);

        result.success = true;
        result.summary = std::move(entry);
    } catch (const std::exception& ex) {
        result.summary = nlohmann::json{
            {"sensor_id", sensor_id}, {"status", "calibration_failed"}, {"error", ex.what()}};
    }
    return result;
}

void collect_gating_flags(const std::vector<PlanarDetections>& detections, bool& has_synth,
                          bool& has_recorded) {
    has_synth = false;
    has_recorded = false;
    for (const auto& det : detections) {
        if (det.tags.count("synthetic")) {
            has_synth = true;
        }
        if (det.tags.count("recorded")) {
            has_recorded = true;
        }
    }
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
    PlanarIntrinsicCalibrationFacade facade;

    bool overall_success = true;
    nlohmann::json cameras = nlohmann::json::array();
    for (const auto& detections : context.dataset.planar_cameras) {
        auto sensor_result = calibrate_sensor(facade, cfg, detections, context);
        cameras.push_back(std::move(sensor_result.summary));
        overall_success = overall_success && sensor_result.success;
    }

    bool has_synthetic = false;
    bool has_recorded = false;
    collect_gating_flags(context.dataset.planar_cameras, has_synthetic, has_recorded);

    result.summary["cameras"] = std::move(cameras);
    result.summary["gating"] = {{"synthetic", has_synthetic}, {"recorded", has_recorded}};
    result.success = overall_success && !context.intrinsic_results.empty();
    return result;
}

}  // namespace calib::pipeline
