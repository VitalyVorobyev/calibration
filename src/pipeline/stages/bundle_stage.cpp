#include <nlohmann/json.hpp>

#include "calib/pipeline/handeye.h"
#include "calib/pipeline/stages.h"
#include "detail/bundle_utils.h"
#include "stages/detail/planar_utils.h"

namespace calib::pipeline {

auto BundleAdjustmentStage::run(PipelineContext& context) -> PipelineStageResult {
    PipelineStageResult result;
    result.name = name();

    if (context.intrinsic_results.empty()) {
        result.summary["status"] = "waiting_for_intrinsic_stage";
        result.success = false;
        return result;
    }
    if (!context.has_bundle_config()) {
        result.summary["status"] = "missing_config";
        result.success = false;
        return result;
    }

    const auto& cfg = context.bundle_config();
    if (cfg.rigs.empty()) {
        result.summary["status"] = "no_rigs_configured";
        result.success = false;
        return result;
    }

    const auto sensor_index = detail::build_sensor_index(context.dataset.planar_cameras);

    context.bundle_results.clear();
    if (!context.artifacts.is_object()) {
        context.artifacts = nlohmann::json::object();
    }
    auto& bundle_artifacts = context.artifacts["bundle"];
    if (!bundle_artifacts.is_object()) {
        bundle_artifacts = nlohmann::json::object();
    }

    const HandEyePipelineConfig* handeye_cfg_ptr =
        context.has_handeye_config() ? &context.handeye_config() : nullptr;

    bool overall_success = true;
    bool any_success = false;
    nlohmann::json rigs_json = nlohmann::json::array();

    for (const auto& rig : cfg.rigs) {
        nlohmann::json rig_json;
        rig_json["rig_id"] = rig.rig_id;
        rig_json["sensor_count"] = rig.sensors.size();
        rig_json["min_angle_deg"] = rig.min_angle_deg;

        const auto* observations = detail::select_bundle_observations(rig, handeye_cfg_ptr);
        const std::size_t requested_views = observations != nullptr ? observations->size() : 0U;

        if (observations == nullptr || observations->empty()) {
            rig_json["status"] = "no_observations";
            rig_json["observations"] =
                nlohmann::json::object({{"requested", requested_views}, {"used", 0}});
            rigs_json.push_back(std::move(rig_json));
            overall_success = false;
            continue;
        }

        auto& rig_artifact = bundle_artifacts[rig.rig_id];
        if (!rig_artifact.is_object()) {
            rig_artifact = nlohmann::json::object();
        }
        rig_artifact["options"] = rig.options;
        rig_artifact["min_angle_deg"] = rig.min_angle_deg;

        const auto sensor_setup =
            detail::collect_bundle_sensor_setup(rig, context.intrinsic_results);
        if (!sensor_setup.missing_sensors.empty() ||
            sensor_setup.cameras.size() != rig.sensors.size()) {
            rig_json["status"] = "missing_intrinsics";
            rig_json["observations"] =
                nlohmann::json::object({{"requested", requested_views}, {"used", 0}});
            rigs_json.push_back(std::move(rig_json));
            overall_success = false;
            continue;
        }

        auto view_result = detail::collect_bundle_observations(
            *observations, rig.sensors, sensor_setup.sensor_to_index, sensor_index,
            context.intrinsic_results);

        rig_json["observations"] = nlohmann::json::object(
            {{"requested", requested_views}, {"used", view_result.observations.size()}});
        rig_json["views"] = view_result.views;

        if (view_result.observations.empty()) {
            rig_json["status"] = "no_valid_observations";
            rigs_json.push_back(std::move(rig_json));
            overall_success = false;
            continue;
        }

        auto handeye_init = detail::compute_handeye_initialization(rig, context.handeye_results,
                                                                   view_result.accumulators);
        rig_json["handeye_initialization"] = handeye_init.report;

        auto target_init =
            detail::choose_initial_target(rig, view_result.accumulators, handeye_init.transforms);
        rig_json["initial_target_source"] = target_init.source;

        rig_artifact["initial_hand_eye"] = handeye_init.report;
        rig_artifact["initial_target"] = target_init.pose;

        if (handeye_init.failed && !rig.initial_target.has_value()) {
            overall_success = false;
        }

        BundleOptions options = rig.options;
        try {
            auto bundle_result =
                optimize_bundle(view_result.observations, sensor_setup.cameras,
                                handeye_init.transforms, target_init.pose, options);

            nlohmann::json result_json;
            result_json["success"] = bundle_result.success;
            result_json["final_cost"] = bundle_result.final_cost;
            result_json["report"] = bundle_result.report;
            result_json["b_se3_t"] = bundle_result.b_se3_t;
            result_json["g_se3_c"] = bundle_result.g_se3_c;
            result_json["cameras"] = bundle_result.cameras;
            if (bundle_result.covariance.size() > 0) {
                result_json["covariance"] = bundle_result.covariance;
            }

            rig_artifact["result"] = result_json;
            rig_artifact["views"] = rig_json["views"];

            rig_json["success"] = bundle_result.success;
            rig_json["final_cost"] = bundle_result.final_cost;

            if (bundle_result.success) {
                rig_json["status"] = "ok";
                any_success = true;
                context.bundle_results[rig.rig_id] = bundle_result;
            } else {
                rig_json["status"] = "optimization_failed";
                overall_success = false;
            }
        } catch (const std::exception& ex) {
            rig_json["status"] = "optimization_error";
            rig_json["error"] = ex.what();
            rig_artifact["error"] = ex.what();
            overall_success = false;
        }

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
