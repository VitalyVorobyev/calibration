#include "calib/pipeline/stages.h"

#include "calib/estimation/linear/handeye.h"
#include "calib/pipeline/handeye.h"
#include "stages/detail/planar_utils.h"

#include <nlohmann/json.hpp>

namespace calib::pipeline {

namespace {

using detail::average_isometries;
using detail::build_sensor_index;
using detail::find_handeye_rig;
using detail::make_planar_view;

struct BundleBuildResult {
    std::vector<PinholeCamera<BrownConradyd>> cameras;
    std::vector<BundleObservation> observations;
    std::vector<std::unordered_map<std::string, std::size_t>> sensor_to_index;
};

}  // namespace

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

    const auto sensor_index = build_sensor_index(context.dataset.planar_cameras);

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

        const std::vector<HandEyeObservationConfig>* observations = nullptr;
        if (!rig.observations.empty()) {
            observations = &rig.observations;
        } else if (handeye_cfg_ptr != nullptr) {
            if (const auto* he_rig = find_handeye_rig(*handeye_cfg_ptr, rig.rig_id)) {
                observations = &he_rig->observations;
            }
        }

        if (observations == nullptr || observations->empty()) {
            rig_json["status"] = "no_observations";
            rig_json["observations"] = nlohmann::json::object({{"requested", 0}, {"used", 0}});
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

        std::unordered_map<std::string, std::size_t> sensor_to_index;
        std::vector<PinholeCamera<BrownConradyd>> cameras;
        cameras.reserve(rig.sensors.size());
        bool sensor_missing = false;
        for (std::size_t idx = 0; idx < rig.sensors.size(); ++idx) {
            const auto& sensor_id = rig.sensors[idx];
            const auto intrinsics_it = context.intrinsic_results.find(sensor_id);
            if (intrinsics_it == context.intrinsic_results.end()) {
                sensor_missing = true;
                break;
            }
            cameras.push_back(intrinsics_it->second.outputs.refine_result.camera);
            sensor_to_index.emplace(sensor_id, idx);
        }
        if (sensor_missing) {
            rig_json["status"] = "missing_intrinsics";
            rigs_json.push_back(std::move(rig_json));
            overall_success = false;
            continue;
        }

        struct SensorAccum {
            std::vector<Eigen::Isometry3d> base;
            std::vector<Eigen::Isometry3d> cam;
        };
        std::vector<SensorAccum> accumulators(rig.sensors.size());

        std::vector<BundleObservation> bundle_observations;
        bundle_observations.reserve(observations->size() * rig.sensors.size());
        nlohmann::json view_reports = nlohmann::json::array();

        for (const auto& view_cfg : *observations) {
            nlohmann::json view_json;
            if (!view_cfg.view_id.empty()) {
                view_json["id"] = view_cfg.view_id;
            }
            view_json["base_pose"] = view_cfg.base_se3_gripper;

            nlohmann::json sensor_reports = nlohmann::json::array();
            bool view_used = false;

            for (const auto& sensor_id : rig.sensors) {
                nlohmann::json sensor_entry;
                sensor_entry["sensor_id"] = sensor_id;

                const auto sensor_idx_it = sensor_to_index.find(sensor_id);
                if (sensor_idx_it == sensor_to_index.end()) {
                    sensor_entry["status"] = "sensor_not_configured";
                    sensor_reports.push_back(std::move(sensor_entry));
                    continue;
                }
                const std::size_t sensor_idx = sensor_idx_it->second;

                const auto image_it = view_cfg.images.find(sensor_id);
                if (image_it == view_cfg.images.end()) {
                    sensor_entry["status"] = "missing_image_reference";
                    sensor_reports.push_back(std::move(sensor_entry));
                    continue;
                }

                const auto det_index_it = sensor_index.find(sensor_id);
                if (det_index_it == sensor_index.end()) {
                    sensor_entry["status"] = "missing_detections";
                    sensor_reports.push_back(std::move(sensor_entry));
                    continue;
                }

                const auto image_lookup_it = det_index_it->second.image_lookup.find(image_it->second);
                if (image_lookup_it == det_index_it->second.image_lookup.end()) {
                    sensor_entry["status"] = "image_not_in_dataset";
                    sensor_entry["image"] = image_it->second;
                    sensor_reports.push_back(std::move(sensor_entry));
                    continue;
                }

                const auto intrinsics_it = context.intrinsic_results.find(sensor_id);
                const auto& camera = intrinsics_it->second.outputs.refine_result.camera;
                const auto* image_det = image_lookup_it->second;

                auto planar_view = make_planar_view(*image_det, intrinsics_it->second.outputs);
                sensor_entry["image"] = image_it->second;
                sensor_entry["points"] = planar_view.size();

                if (planar_view.size() < 4U) {
                    sensor_entry["status"] = "insufficient_points";
                    sensor_reports.push_back(std::move(sensor_entry));
                    continue;
                }

                Eigen::Isometry3d cam_se3_target = estimate_planar_pose(planar_view, camera);

                BundleObservation obs;
                obs.view = planar_view;
                obs.b_se3_g = view_cfg.base_se3_gripper;
                obs.camera_index = sensor_idx;
                bundle_observations.push_back(std::move(obs));

                accumulators[sensor_idx].base.push_back(view_cfg.base_se3_gripper);
                accumulators[sensor_idx].cam.push_back(cam_se3_target);

                sensor_entry["status"] = "ok";
                view_used = true;
                sensor_reports.push_back(std::move(sensor_entry));
            }

            view_json["sensors"] = std::move(sensor_reports);
            view_json["used"] = view_used;
            view_reports.push_back(std::move(view_json));
        }

        rig_json["observations"] = nlohmann::json{{"requested", observations->size()},
                                                  {"used", bundle_observations.size()}};
        rig_json["views"] = view_reports;

        if (bundle_observations.empty()) {
            rig_json["status"] = "no_valid_observations";
            rigs_json.push_back(std::move(rig_json));
            overall_success = false;
            continue;
        }

        std::vector<Eigen::Isometry3d> init_g_se3_c(rig.sensors.size(), Eigen::Isometry3d::Identity());
        nlohmann::json handeye_init = nlohmann::json::array();
        const auto rig_handeye_it = context.handeye_results.find(rig.rig_id);
        bool init_failure = false;

        for (std::size_t idx = 0; idx < rig.sensors.size(); ++idx) {
            const auto& sensor_id = rig.sensors[idx];
            nlohmann::json init_entry;
            init_entry["sensor_id"] = sensor_id;
            init_entry["source"] = "identity";

            if (rig_handeye_it != context.handeye_results.end()) {
                const auto& sensor_map = rig_handeye_it->second;
                const auto he_it = sensor_map.find(sensor_id);
                if (he_it != sensor_map.end() && he_it->second.success) {
                    init_g_se3_c[idx] = he_it->second.g_se3_c;
                    init_entry["source"] = "handeye";
                    init_entry["success"] = true;
                    handeye_init.push_back(std::move(init_entry));
                    continue;
                }
            }

            if (accumulators[idx].cam.size() >= 2U) {
                try {
                    init_g_se3_c[idx] =
                        estimate_handeye_dlt(accumulators[idx].base, accumulators[idx].cam, rig.min_angle_deg);
                    init_entry["source"] = "dlt";
                    init_entry["success"] = true;
                } catch (const std::exception& ex) {
                    init_entry["source"] = "dlt";
                    init_entry["success"] = false;
                    init_entry["error"] = ex.what();
                    init_failure = true;
                }
            } else {
                init_entry["success"] = false;
                init_entry["error"] = "insufficient_observations";
                init_failure = true;
            }
            handeye_init.push_back(std::move(init_entry));
        }

        rig_json["handeye_initialization"] = handeye_init;

        Eigen::Isometry3d init_b_se3_t = Eigen::Isometry3d::Identity();
        std::vector<Eigen::Isometry3d> target_candidates;

        if (rig.initial_target.has_value()) {
            init_b_se3_t = *rig.initial_target;
            rig_json["initial_target_source"] = "config";
        } else {
            for (std::size_t idx = 0; idx < accumulators.size(); ++idx) {
                const auto& acc = accumulators[idx];
                const auto& g_pose = init_g_se3_c[idx];
                for (std::size_t obs_idx = 0; obs_idx < acc.base.size(); ++obs_idx) {
                    const auto& base_pose = acc.base[obs_idx];
                    const auto& cam_pose = acc.cam[obs_idx];
                    target_candidates.push_back(base_pose * g_pose * cam_pose);
                }
            }
            if (!target_candidates.empty()) {
                init_b_se3_t = average_isometries(target_candidates);
                rig_json["initial_target_source"] = "estimated";
            } else {
                rig_json["initial_target_source"] = "identity";
            }
        }

        rig_artifact["initial_hand_eye"] = handeye_init;
        rig_artifact["initial_target"] = init_b_se3_t;

        if (init_failure && !rig.initial_target.has_value()) {
            overall_success = false;
        }

        BundleOptions options = rig.options;
        try {
            auto bundle_result =
                optimize_bundle(bundle_observations, cameras, init_g_se3_c, init_b_se3_t, options);

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
