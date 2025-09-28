#include "calib/pipeline/stages.h"

#include "calib/estimation/linear/handeye.h"
#include "calib/pipeline/handeye.h"
#include "stages/detail/planar_utils.h"

#include <nlohmann/json.hpp>

namespace calib::pipeline {

namespace {

using detail::build_sensor_index;
using detail::make_planar_view;

struct SensorAccumulators {
    std::vector<Eigen::Isometry3d> base;
    std::vector<Eigen::Isometry3d> cam;
};

void ensure_artifact_slot(nlohmann::json& artifacts, const std::string& rig_id,
                          double min_angle, const HandeyeOptions& opts) {
    auto& rig_artifact = artifacts[rig_id];
    if (!rig_artifact.is_object()) {
        rig_artifact = nlohmann::json::object();
    }
    rig_artifact["min_angle_deg"] = min_angle;
    rig_artifact["options"] = opts;
    auto& sensors = rig_artifact["sensors"];
    if (!sensors.is_object()) {
        sensors = nlohmann::json::object();
    }
}

}  // namespace

auto HandEyeCalibrationStage::run(PipelineContext& context) -> PipelineStageResult {
    PipelineStageResult result;
    result.name = name();

    if (context.intrinsic_results.empty()) {
        result.summary["status"] = "waiting_for_intrinsic_stage";
        result.success = false;
        return result;
    }
    if (!context.has_handeye_config()) {
        result.summary["status"] = "missing_config";
        result.success = false;
        return result;
    }

    const auto& cfg = context.handeye_config();
    if (cfg.rigs.empty()) {
        result.summary["status"] = "no_rigs_configured";
        result.success = false;
        return result;
    }

    const auto sensor_index = build_sensor_index(context.dataset.planar_cameras);

    context.handeye_results.clear();
    if (!context.artifacts.is_object()) {
        context.artifacts = nlohmann::json::object();
    }
    auto& handeye_artifacts = context.artifacts["hand_eye"];
    if (!handeye_artifacts.is_object()) {
        handeye_artifacts = nlohmann::json::object();
    }

    bool overall_success = true;
    bool any_success = false;
    nlohmann::json rigs_json = nlohmann::json::array();

    for (const auto& rig : cfg.rigs) {
        nlohmann::json rig_json;
        rig_json["rig_id"] = rig.rig_id;
        rig_json["sensor_count"] = rig.sensors.size();
        rig_json["min_angle_deg"] = rig.min_angle_deg;

        nlohmann::json sensors_json = nlohmann::json::array();
        bool rig_success = true;
        bool rig_any_sensor = false;

        ensure_artifact_slot(handeye_artifacts, rig.rig_id, rig.min_angle_deg, rig.options);
        auto& sensors_artifact = handeye_artifacts[rig.rig_id]["sensors"];

        for (const auto& sensor_id : rig.sensors) {
            nlohmann::json sensor_json;
            sensor_json["sensor_id"] = sensor_id;
            sensor_json["requested_observations"] = rig.observations.size();
            sensor_json["min_angle_deg"] = rig.min_angle_deg;

            const auto intrinsics_it = context.intrinsic_results.find(sensor_id);
            if (intrinsics_it == context.intrinsic_results.end()) {
                sensor_json["status"] = "missing_intrinsics";
                rig_success = false;
                sensors_json.push_back(sensor_json);
                sensors_artifact[sensor_id] = sensor_json;
                continue;
            }

            const auto index_it = sensor_index.find(sensor_id);
            if (index_it == sensor_index.end()) {
                sensor_json["status"] = "missing_detections";
                rig_success = false;
                sensors_json.push_back(sensor_json);
                sensors_artifact[sensor_id] = sensor_json;
                continue;
            }

            const auto& intrinsics = intrinsics_it->second;
            const auto& detection_index = index_it->second;
            const auto& camera = intrinsics.outputs.refine_result.camera;

            nlohmann::json view_reports = nlohmann::json::array();
            SensorAccumulators accum;

            for (const auto& view_cfg : rig.observations) {
                nlohmann::json view_json;
                if (!view_cfg.view_id.empty()) {
                    view_json["id"] = view_cfg.view_id;
                }
                view_json["base_pose"] = view_cfg.base_se3_gripper;

                const auto image_it = view_cfg.images.find(sensor_id);
                if (image_it == view_cfg.images.end()) {
                    view_json["status"] = "missing_image_reference";
                    view_reports.push_back(std::move(view_json));
                    continue;
                }

                const auto det_it = detection_index.image_lookup.find(image_it->second);
                if (det_it == detection_index.image_lookup.end()) {
                    view_json["status"] = "image_not_in_dataset";
                    view_reports.push_back(std::move(view_json));
                    continue;
                }

            const auto* image_det = det_it->second;
                auto planar_view = make_planar_view(*image_det, intrinsics.outputs);
                view_json["points"] = planar_view.size();

                if (planar_view.size() < 4U) {
                    view_json["status"] = "insufficient_points";
                    view_reports.push_back(std::move(view_json));
                    continue;
                }

                Eigen::Isometry3d cam_se3_target = estimate_planar_pose(planar_view, camera);
                accum.base.push_back(view_cfg.base_se3_gripper);
                accum.cam.push_back(cam_se3_target);
                view_json["status"] = "ok";
                view_reports.push_back(std::move(view_json));
            }

            sensor_json["used_observations"] = accum.cam.size();
            sensor_json["views"] = view_reports;

            if (accum.cam.size() < 2U) {
                sensor_json["status"] = accum.cam.empty() ? "no_observations" : "insufficient_observations";
                rig_success = false;
                sensors_json.push_back(sensor_json);
                sensors_artifact[sensor_id] = sensor_json;
                continue;
            }

            try {
                auto he_result = estimate_and_optimize_handeye(accum.base, accum.cam, rig.min_angle_deg,
                                                               rig.options);
                sensor_json["status"] = he_result.success ? "ok" : "optimization_failed";
                sensor_json["success"] = he_result.success;
                sensor_json["final_cost"] = he_result.final_cost;
                sensor_json["report"] = he_result.report;
                sensor_json["g_se3_c"] = he_result.g_se3_c;
                if (he_result.covariance.size() > 0) {
                    sensor_json["covariance"] = he_result.covariance;
                }

                sensors_artifact[sensor_id] = sensor_json;

                if (he_result.success) {
                    rig_any_sensor = true;
                    context.handeye_results[rig.rig_id][sensor_id] = he_result;
                } else {
                    rig_success = false;
                }
            } catch (const std::exception& ex) {
                sensor_json["status"] = "estimation_error";
                sensor_json["error"] = ex.what();
                sensors_artifact[sensor_id] = sensor_json;
                rig_success = false;
            }

            sensors_json.push_back(sensor_json);
        }

        if (rig_any_sensor && rig_success) {
            rig_json["status"] = "ok";
            any_success = true;
        } else if (rig_any_sensor) {
            rig_json["status"] = "partial_success";
            any_success = true;
            overall_success = false;
        } else {
            rig_json["status"] = "failed";
            overall_success = false;
        }

        rig_json["sensor_reports"] = std::move(sensors_json);
        rigs_json.push_back(std::move(rig_json));
    }

    result.summary["rigs"] = std::move(rigs_json);
    if (any_success && overall_success) {
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
