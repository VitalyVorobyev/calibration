#include "calib/pipeline/extrinsics.h"

// std
#include <Eigen/Geometry>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

#include "calib/io/serialization.h"
#include "calib/models/distortion.h"

namespace calib::pipeline {

namespace {

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

[[nodiscard]] auto to_dual_camera(const PinholeCamera<BrownConradyd>& cam)
    -> PinholeCamera<DualDistortion> {
    DualDistortion dual;
    dual.forward = cam.distortion.coeffs;
    dual.inverse = invert_brown_conrady(cam.distortion.coeffs);
    return PinholeCamera<DualDistortion>(cam.kmtx, dual);
}

}  // namespace

StereoPairConfig::StereoPairConfig() {
    options.optimize_intrinsics = false;
    options.optimize_extrinsics = true;
    options.optimize_skew = false;
}

void to_json(nlohmann::json& j, const StereoViewSelection& view) {
    j = {{"reference_image", view.reference_image}, {"target_image", view.target_image}};
}

void from_json(const nlohmann::json& j, StereoViewSelection& view) {
    j.at("reference_image").get_to(view.reference_image);
    j.at("target_image").get_to(view.target_image);
}

static void extrinsic_options_to_json(nlohmann::json& j, const ExtrinsicOptions& opts) {
    j = {{"optimize_intrinsics", opts.optimize_intrinsics},
         {"optimize_skew", opts.optimize_skew},
         {"optimize_extrinsics", opts.optimize_extrinsics},
         {"huber_delta", opts.huber_delta},
         {"epsilon", opts.epsilon},
         {"max_iterations", opts.max_iterations},
         {"compute_covariance", opts.compute_covariance},
         {"verbose", opts.verbose}};
}

static void extrinsic_options_from_json(const nlohmann::json& j, ExtrinsicOptions& opts) {
    opts.optimize_intrinsics = j.value("optimize_intrinsics", opts.optimize_intrinsics);
    opts.optimize_skew = j.value("optimize_skew", opts.optimize_skew);
    opts.optimize_extrinsics = j.value("optimize_extrinsics", opts.optimize_extrinsics);
    opts.huber_delta = j.value("huber_delta", opts.huber_delta);
    opts.epsilon = j.value("epsilon", opts.epsilon);
    opts.max_iterations = j.value("max_iterations", opts.max_iterations);
    opts.compute_covariance = j.value("compute_covariance", opts.compute_covariance);
    opts.verbose = j.value("verbose", opts.verbose);
}

void to_json(nlohmann::json& j, const StereoPairConfig& cfg) {
    nlohmann::json options_json;
    extrinsic_options_to_json(options_json, cfg.options);
    j = {{"pair_id", cfg.pair_id},
         {"reference_sensor", cfg.reference_sensor},
         {"target_sensor", cfg.target_sensor},
         {"views", cfg.views},
         {"options", options_json}};
}

void from_json(const nlohmann::json& j, StereoPairConfig& cfg) {
    if (j.contains("pair_id")) {
        j.at("pair_id").get_to(cfg.pair_id);
    }
    j.at("reference_sensor").get_to(cfg.reference_sensor);
    j.at("target_sensor").get_to(cfg.target_sensor);
    if (j.contains("views")) {
        j.at("views").get_to(cfg.views);
    }
    if (j.contains("options")) {
        extrinsic_options_from_json(j.at("options"), cfg.options);
    }
    if (cfg.pair_id.empty()) {
        cfg.pair_id = cfg.reference_sensor + "_" + cfg.target_sensor;
    }
}

void to_json(nlohmann::json& j, const StereoCalibrationConfig& cfg) { j = {{"pairs", cfg.pairs}}; }

void from_json(const nlohmann::json& j, StereoCalibrationConfig& cfg) {
    cfg.pairs.clear();
    if (j.contains("pairs")) {
        j.at("pairs").get_to(cfg.pairs);
    }
}

void to_json(nlohmann::json& j, const StereoCalibrationViewSummary& summary) {
    j = {{"reference_image", summary.reference_image},
         {"target_image", summary.target_image},
         {"reference_points", summary.reference_points},
         {"target_points", summary.target_points},
         {"status", summary.status}};
}

[[nodiscard]] auto compute_views(const StereoPairConfig& cfg,
                                 const planar::PlanarDetections& reference_detections,
                                 const planar::PlanarDetections& target_detections,
                                 const planar::CalibrationRunResult& reference_intrinsics,
                                 const planar::CalibrationRunResult& target_intrinsics,
                                 StereoCalibrationRunResult& result) -> std::vector<MulticamPlanarView> {
    const auto reference_lookup = build_point_lookup(reference_detections);
    const auto target_lookup = build_point_lookup(target_detections);

    std::vector<MulticamPlanarView> views;
    views.reserve(cfg.views.size());

    for (const auto& view_cfg : cfg.views) {
        StereoCalibrationViewSummary summary;
        summary.reference_image = view_cfg.reference_image;
        summary.target_image = view_cfg.target_image;

        const auto ref_it = reference_lookup.find(view_cfg.reference_image);
        const auto tgt_it = target_lookup.find(view_cfg.target_image);

        if (ref_it == reference_lookup.end()) {
            summary.status = "missing_reference_image";
            result.view_summaries.push_back(std::move(summary));
            continue;
        }
        if (tgt_it == target_lookup.end()) {
            summary.status = "missing_target_image";
            result.view_summaries.push_back(std::move(summary));
            continue;
        }

        auto reference_view = make_planar_view(*ref_it->second, reference_intrinsics.outputs);
        auto target_view = make_planar_view(*tgt_it->second, target_intrinsics.outputs);
        summary.reference_points = reference_view.size();
        summary.target_points = target_view.size();

        if (reference_view.size() < 4U || target_view.size() < 4U) {
            summary.status = "insufficient_points";
            result.view_summaries.push_back(std::move(summary));
            continue;
        }

        MulticamPlanarView multi_view;
        multi_view.push_back(std::move(reference_view));
        multi_view.push_back(std::move(target_view));
        views.push_back(std::move(multi_view));
        summary.status = "ok";
        result.view_summaries.push_back(std::move(summary));
    }

    return views;
}

auto StereoCalibrationFacade::calibrate(const StereoPairConfig& cfg,
                                        const planar::PlanarDetections& reference_detections,
                                        const planar::PlanarDetections& target_detections,
                                        const planar::CalibrationRunResult& reference_intrinsics,
                                        const planar::CalibrationRunResult& target_intrinsics) const
    -> StereoCalibrationRunResult {
    StereoCalibrationRunResult result;
    result.requested_views = cfg.views.size();

    if (reference_intrinsics.outputs.refine_result.camera.distortion.coeffs.size() == 0 ||
        target_intrinsics.outputs.refine_result.camera.distortion.coeffs.size() == 0) {
        throw std::runtime_error("StereoCalibrationFacade: camera intrinsics are not available.");
    }

    const auto views = compute_views(cfg, reference_detections, target_detections,
                                     reference_intrinsics, target_intrinsics, result);

    result.used_views = views.size();
    if (views.empty()) {
        result.success = false;
        result.optimization.success = false;
        return result;
    }

    std::vector<PinholeCamera<BrownConradyd>> init_cameras = {
        reference_intrinsics.outputs.refine_result.camera,
        target_intrinsics.outputs.refine_result.camera};

    std::vector<PinholeCamera<DualDistortion>> dlt_cameras;
    dlt_cameras.reserve(init_cameras.size());
    std::transform(init_cameras.begin(), init_cameras.end(), std::back_inserter(dlt_cameras),
                   [](const auto& cam) { return to_dual_camera(cam); });

    result.initial_guess = estimate_extrinsic_dlt(views, dlt_cameras);

    ExtrinsicOptions options = cfg.options;
    options.optimize_intrinsics = cfg.options.optimize_intrinsics;

    result.optimization = optimize_extrinsics(views, init_cameras, result.initial_guess.c_se3_r,
                                              result.initial_guess.r_se3_t, options);
    result.success = result.optimization.success;
    return result;
}

// ---- Multicam generalization implementations ----

void to_json(nlohmann::json& j, const MultiCameraViewSelection& view) { j = view.images; }

void from_json(const nlohmann::json& j, MultiCameraViewSelection& view) {
    view.images.clear();
    for (auto it = j.begin(); it != j.end(); ++it) {
        view.images.emplace(it.key(), it.value().get<std::string>());
    }
}

void to_json(nlohmann::json& j, const MultiCameraRigConfig& cfg) {
    nlohmann::json options_json;
    extrinsic_options_to_json(options_json, cfg.options);
    j = {{"rig_id", cfg.rig_id}, {"sensors", cfg.sensors}, {"views", cfg.views}, {"options", options_json}};
}

void from_json(const nlohmann::json& j, MultiCameraRigConfig& cfg) {
    if (j.contains("rig_id")) j.at("rig_id").get_to(cfg.rig_id);
    j.at("sensors").get_to(cfg.sensors);
    if (j.contains("views")) j.at("views").get_to(cfg.views);
    if (j.contains("options")) extrinsic_options_from_json(j.at("options"), cfg.options);
    if (cfg.rig_id.empty()) {
        cfg.rig_id = cfg.sensors.empty() ? std::string{"rig"} : cfg.sensors.front();
    }
}

static auto compute_views(const MultiCameraRigConfig& cfg,
                          const std::unordered_map<std::string, planar::PlanarDetections>& dets,
                          const std::unordered_map<std::string, planar::CalibrationRunResult>& intr,
                          MultiCameraCalibrationRunResult& /*result*/)
    -> std::vector<MulticamPlanarView> {
    // Build lookup tables for each sensor
    std::unordered_map<std::string, std::unordered_map<std::string, const planar::PlanarImageDetections*>>
        lookup;
    for (const auto& [sid, d] : dets) {
        std::unordered_map<std::string, const planar::PlanarImageDetections*> map;
        for (const auto& img : d.images) map.emplace(img.file, &img);
        lookup.emplace(sid, std::move(map));
    }

    std::vector<MulticamPlanarView> views;
    views.reserve(cfg.views.size());
    for (const auto& view_sel : cfg.views) {
        MulticamPlanarView multi;
        multi.resize(cfg.sensors.size());
        bool ok = true;
        for (std::size_t i = 0; i < cfg.sensors.size(); ++i) {
            const auto& sid = cfg.sensors[i];
            auto img_it = view_sel.images.find(sid);
            if (img_it == view_sel.images.end()) {
                ok = false;
                break;
            }
            auto det_cam_it = dets.find(sid);
            if (det_cam_it == dets.end()) {
                ok = false;
                break;
            }
            const auto& img_lookup = lookup.at(sid);
            auto img_det_it = img_lookup.find(img_it->second);
            if (img_det_it == img_lookup.end()) {
                ok = false;
                break;
            }
            auto view = make_planar_view(*img_det_it->second, intr.at(sid).outputs);
            if (view.size() < 4U) {
                ok = false;
                break;
            }
            multi[i] = std::move(view);
        }
        if (ok) {
            views.push_back(std::move(multi));
        }
    }
    return views;
}

auto MultiCameraCalibrationFacade::calibrate(
    const MultiCameraRigConfig& cfg,
    const std::unordered_map<std::string, planar::PlanarDetections>& detections_by_sensor,
    const std::unordered_map<std::string, planar::CalibrationRunResult>& intrinsics_by_sensor) const
    -> MultiCameraCalibrationRunResult {
    MultiCameraCalibrationRunResult result;
    result.requested_views = cfg.views.size();
    result.sensors = cfg.sensors;

    // Validate intrinsics availability
    for (const auto& sid : cfg.sensors) {
        auto it = intrinsics_by_sensor.find(sid);
        if (it == intrinsics_by_sensor.end() ||
            it->second.outputs.refine_result.camera.distortion.coeffs.size() == 0) {
            throw std::runtime_error("MultiCameraCalibrationFacade: intrinsics not available for sensor: " + sid);
        }
    }

    const auto views = compute_views(cfg, detections_by_sensor, intrinsics_by_sensor, result);
    result.used_views = views.size();
    if (views.empty()) {
        result.success = false;
        result.optimization.success = false;
        return result;
    }

    std::vector<PinholeCamera<BrownConradyd>> init_cameras;
    init_cameras.reserve(cfg.sensors.size());
    for (const auto& sid : cfg.sensors) {
        init_cameras.push_back(intrinsics_by_sensor.at(sid).outputs.refine_result.camera);
    }

    std::vector<PinholeCamera<DualDistortion>> dlt_cameras;
    dlt_cameras.reserve(init_cameras.size());
    std::transform(init_cameras.begin(), init_cameras.end(), std::back_inserter(dlt_cameras),
                   [](const auto& cam) { return to_dual_camera(cam); });

    result.initial_guess = estimate_extrinsic_dlt(views, dlt_cameras);

    ExtrinsicOptions options = cfg.options;
    result.optimization = optimize_extrinsics(views, init_cameras, result.initial_guess.c_se3_r,
                                              result.initial_guess.r_se3_t, options);
    result.success = result.optimization.success;
    return result;
}

}  // namespace calib::pipeline
