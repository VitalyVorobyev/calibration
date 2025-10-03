#include "calib/pipeline/detail/bundle_utils.h"

#include <algorithm>
#include <stdexcept>

#include "calib/estimation/linear/handeye.h"
#include "calib/estimation/linear/planarpose.h"

namespace calib::pipeline::detail {

auto collect_bundle_sensor_setup(
    const BundleRigConfig& rig,
    const std::unordered_map<std::string, IntrinsicCalibrationOutputs>& intrinsics)
    -> BundleSensorSetup {
    BundleSensorSetup setup;
    setup.cameras.reserve(rig.sensors.size());
    for (std::size_t idx = 0; idx < rig.sensors.size(); ++idx) {
        const auto& sensor_id = rig.sensors[idx];
        const auto intrinsics_it = intrinsics.find(sensor_id);
        if (intrinsics_it == intrinsics.end()) {
            setup.missing_sensors.push_back(sensor_id);
            continue;
        }
        setup.sensor_to_index.emplace(sensor_id, idx);
        setup.cameras.push_back(intrinsics_it->second.refine_result.camera);
    }
    return setup;
}

const std::vector<HandEyeObservationConfig>* select_bundle_observations(
    const BundleRigConfig& rig, const HandEyePipelineConfig* handeye_cfg) {
    if (!rig.observations.empty()) {
        return &rig.observations;
    }
    if (handeye_cfg == nullptr) {
        return nullptr;
    }
    if (const auto* he_rig = find_handeye_rig(*handeye_cfg, rig.rig_id)) {
        if (!he_rig->observations.empty()) {
            return &he_rig->observations;
        }
    }
    return nullptr;
}

auto collect_bundle_observations(
    const std::vector<HandEyeObservationConfig>& observation_config,
    const std::vector<std::string>& sensors,
    const std::unordered_map<std::string, std::size_t>& sensor_to_index,
    const std::unordered_map<std::string, SensorDetectionsIndex>& sensor_index,
    const std::unordered_map<std::string, IntrinsicCalibrationOutputs>& intrinsics)
    -> BundleViewProcessingResult {
    BundleViewProcessingResult output;
    output.views = nlohmann::json::array();
    output.accumulators.resize(sensors.size());

    for (const auto& view_cfg : observation_config) {
        nlohmann::json view_json;
        if (!view_cfg.view_id.empty()) {
            view_json["id"] = view_cfg.view_id;
        }
        view_json["base_pose"] = view_cfg.base_se3_gripper;

        nlohmann::json sensor_reports = nlohmann::json::array();
        bool view_used = false;

        for (const auto& sensor_id : sensors) {
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

            const auto lookup_it = det_index_it->second.image_lookup.find(image_it->second);
            if (lookup_it == det_index_it->second.image_lookup.end()) {
                sensor_entry["status"] = "image_not_in_dataset";
                sensor_entry["image"] = image_it->second;
                sensor_reports.push_back(std::move(sensor_entry));
                continue;
            }

            const auto intrinsics_it = intrinsics.find(sensor_id);
            if (intrinsics_it == intrinsics.end()) {
                sensor_entry["status"] = "missing_intrinsics";
                sensor_reports.push_back(std::move(sensor_entry));
                continue;
            }
            const auto& camera = intrinsics_it->second.refine_result.camera;

            const auto* image_det = lookup_it->second;
            auto planar_view = make_planar_view(*image_det);

            sensor_entry["image"] = image_it->second;
            sensor_entry["points"] = planar_view.size();

            if (planar_view.size() < 4U) {
                sensor_entry["status"] = "insufficient_points";
                sensor_reports.push_back(std::move(sensor_entry));
                continue;
            }

            Eigen::Isometry3d cam_se3_target = estimate_planar_pose(planar_view, camera);

            BundleObservation obs;
            obs.view = std::move(planar_view);
            obs.b_se3_g = view_cfg.base_se3_gripper;
            obs.camera_index = sensor_idx;
            output.observations.push_back(std::move(obs));

            output.accumulators[sensor_idx].base.push_back(view_cfg.base_se3_gripper);
            output.accumulators[sensor_idx].cam.push_back(cam_se3_target);

            sensor_entry["status"] = "ok";
            view_used = true;
            sensor_reports.push_back(std::move(sensor_entry));
        }

        view_json["sensors"] = std::move(sensor_reports);
        view_json["used"] = view_used;
        output.views.push_back(std::move(view_json));
        if (view_used) {
            ++output.used_view_count;
        }
    }

    return output;
}

auto compute_handeye_initialization(
    const BundleRigConfig& rig,
    const std::unordered_map<std::string, std::unordered_map<std::string, HandeyeResult>>&
        handeye_results,
    const std::vector<SensorAccumulator>& accumulators) -> HandeyeInitializationResult {
    HandeyeInitializationResult output;
    output.report = nlohmann::json::array();
    output.transforms.assign(rig.sensors.size(), Eigen::Isometry3d::Identity());

    const auto rig_handeye_it = handeye_results.find(rig.rig_id);
    const auto* sensor_map =
        rig_handeye_it != handeye_results.end() ? &rig_handeye_it->second : nullptr;

    for (std::size_t idx = 0; idx < rig.sensors.size(); ++idx) {
        const auto& sensor_id = rig.sensors[idx];
        nlohmann::json entry;
        entry["sensor_id"] = sensor_id;
        entry["source"] = "identity";

        if (sensor_map != nullptr) {
            const auto he_it = sensor_map->find(sensor_id);
            if (he_it != sensor_map->end() && he_it->second.core.success) {
                output.transforms[idx] = he_it->second.g_se3_c;
                entry["source"] = "handeye";
                entry["success"] = true;
                output.report.push_back(std::move(entry));
                continue;
            }
        }

        if (idx < accumulators.size() && accumulators[idx].cam.size() >= 2U) {
            try {
                output.transforms[idx] = estimate_handeye_dlt(
                    accumulators[idx].base, accumulators[idx].cam, rig.min_angle_deg);
                entry["source"] = "dlt";
                entry["success"] = true;
            } catch (const std::exception& ex) {
                entry["source"] = "dlt";
                entry["success"] = false;
                entry["error"] = ex.what();
                output.failed = true;
            }
        } else {
            entry["success"] = false;
            entry["error"] = "insufficient_observations";
            output.failed = true;
        }

        output.report.push_back(std::move(entry));
    }

    return output;
}

auto choose_initial_target(const BundleRigConfig& rig,
                           const std::vector<SensorAccumulator>& accumulators,
                           const std::vector<Eigen::Isometry3d>& init_g_se3_c)
    -> TargetInitializationResult {
    TargetInitializationResult output;

    if (rig.initial_target.has_value()) {
        output.pose = *rig.initial_target;
        output.source = "config";
        return output;
    }

    std::vector<Eigen::Isometry3d> candidates;
    for (std::size_t idx = 0; idx < accumulators.size(); ++idx) {
        const auto& acc = accumulators[idx];
        if (idx >= init_g_se3_c.size()) {
            continue;
        }
        const auto& g_pose = init_g_se3_c[idx];
        for (std::size_t obs_idx = 0; obs_idx < acc.base.size(); ++obs_idx) {
            const auto& base_pose = acc.base[obs_idx];
            const auto& cam_pose = acc.cam[obs_idx];
            candidates.push_back(base_pose * g_pose * cam_pose);
        }
    }

    if (!candidates.empty()) {
        output.pose = average_isometries(candidates);
        output.source = "estimated";
    } else {
        output.pose = Eigen::Isometry3d::Identity();
        output.source = "identity";
    }

    return output;
}

}  // namespace calib::pipeline::detail
