#include "calib/pipeline/stages.h"

#include "calib/estimation/linear/handeye.h"
#include "calib/io/serialization.h"
#include "calib/pipeline/extrinsics.h"
#include "calib/pipeline/handeye.h"

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

struct SensorDetectionsIndex final {
    const planar::PlanarDetections* detections{nullptr};
    std::unordered_map<std::string, const planar::PlanarImageDetections*> image_lookup;
};

[[nodiscard]] auto build_point_lookup(const planar::PlanarDetections& detections)
    -> std::unordered_map<std::string, const planar::PlanarImageDetections*> {
    std::unordered_map<std::string, const planar::PlanarImageDetections*> lookup;
    for (const auto& image : detections.images) {
        lookup.emplace(image.file, &image);
    }
    return lookup;
}

[[nodiscard]] auto make_planar_view(const planar::PlanarImageDetections& detections,
                                    const planar::CalibrationOutputs& outputs) -> PlanarView {
    PlanarView view;
    view.reserve(detections.points.size());
    for (const auto& point : detections.points) {
        PlanarObservation obs;
        obs.object_xy =
            Eigen::Vector2d((point.local_x - outputs.point_center[0]) * outputs.point_scale,
                            (point.local_y - outputs.point_center[1]) * outputs.point_scale);
        obs.image_uv = Eigen::Vector2d(point.x, point.y);
        view.push_back(std::move(obs));
    }
    return view;
}

[[nodiscard]] auto build_sensor_index(const std::vector<planar::PlanarDetections>& detections)
    -> std::unordered_map<std::string, SensorDetectionsIndex> {
    std::unordered_map<std::string, SensorDetectionsIndex> index;
    for (const auto& det : detections) {
        if (det.sensor_id.empty()) {
            continue;
        }
        SensorDetectionsIndex entry;
        entry.detections = &det;
        entry.image_lookup = build_point_lookup(det);
        index.emplace(det.sensor_id, std::move(entry));
    }
    return index;
}

[[nodiscard]] auto average_isometries(const std::vector<Eigen::Isometry3d>& poses)
    -> Eigen::Isometry3d {
    if (poses.empty()) {
        return Eigen::Isometry3d::Identity();
    }
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat_sum(0.0, 0.0, 0.0, 0.0);
    for (const auto& pose : poses) {
        translation += pose.translation();
        Eigen::Quaterniond q(pose.linear());
        if (quat_sum.coeffs().dot(q.coeffs()) < 0.0) {
            q.coeffs() *= -1.0;
        }
        quat_sum.coeffs() += q.coeffs();
    }
    translation /= static_cast<double>(poses.size());
    quat_sum.normalize();
    Eigen::Isometry3d avg = Eigen::Isometry3d::Identity();
    avg.linear() = quat_sum.toRotationMatrix();
    avg.translation() = translation;
    return avg;
}

[[nodiscard]] auto find_handeye_rig(const HandEyePipelineConfig& cfg, const std::string& rig_id)
    -> const HandEyeRigConfig* {
    const auto it = std::find_if(cfg.rigs.begin(), cfg.rigs.end(),
                                 [&](const HandEyeRigConfig& rig) { return rig.rig_id == rig_id; });
    return it == cfg.rigs.end() ? nullptr : &(*it);
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
                nlohmann::json view_json = view;
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

        auto& rig_artifact = handeye_artifacts[rig.rig_id];
        if (!rig_artifact.is_object()) {
            rig_artifact = nlohmann::json::object();
        }
        rig_artifact["min_angle_deg"] = rig.min_angle_deg;
        rig_artifact["options"] = rig.options;
        auto& sensors_artifact = rig_artifact["sensors"];
        if (!sensors_artifact.is_object()) {
            sensors_artifact = nlohmann::json::object();
        }

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

            struct SensorAccum {
                std::vector<Eigen::Isometry3d> base;
                std::vector<Eigen::Isometry3d> cam;
            } accum;

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

                const auto& image_name = image_it->second;
                view_json["image"] = image_name;

                const auto det_it = detection_index.image_lookup.find(image_name);
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
                sensor_json["status"] =
                    accum.cam.empty() ? "no_observations" : "insufficient_observations";
                rig_success = false;
                sensors_json.push_back(sensor_json);
                sensors_artifact[sensor_id] = sensor_json;
                continue;
            }

            try {
                auto he_result = estimate_and_optimize_handeye(accum.base, accum.cam,
                                                               rig.min_angle_deg, rig.options);
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
                rig_success = false;
                sensors_artifact[sensor_id] = sensor_json;
            }

            sensors_json.push_back(sensor_json);
        }

        rig_json["sensor_reports"] = std::move(sensors_json);

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
            const auto* he_rig = find_handeye_rig(*handeye_cfg_ptr, rig.rig_id);
            if (he_rig != nullptr) {
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
        for (std::size_t idx = 0; idx < rig.sensors.size(); ++idx) {
            sensor_to_index.emplace(rig.sensors[idx], idx);
        }

        std::vector<PinholeCamera<BrownConradyd>> cameras;
        cameras.reserve(rig.sensors.size());
        bool sensor_missing = false;
        for (const auto& sensor_id : rig.sensors) {
            const auto intrinsics_it = context.intrinsic_results.find(sensor_id);
            if (intrinsics_it == context.intrinsic_results.end()) {
                sensor_missing = true;
                break;
            }
            cameras.push_back(intrinsics_it->second.outputs.refine_result.camera);
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

                const auto image_lookup_it =
                    det_index_it->second.image_lookup.find(image_it->second);
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

        std::vector<Eigen::Isometry3d> init_g_se3_c(rig.sensors.size(),
                                                    Eigen::Isometry3d::Identity());
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
                    init_g_se3_c[idx] = estimate_handeye_dlt(
                        accumulators[idx].base, accumulators[idx].cam, rig.min_angle_deg);
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
                    target_candidates.push_back(base_pose * g_pose * cam_pose.inverse());
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
