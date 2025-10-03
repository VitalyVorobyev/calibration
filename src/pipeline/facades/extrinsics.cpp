#include "calib/pipeline/facades/extrinsics.h"

// std
#include <Eigen/Geometry>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

#include "calib/io/serialization.h"
#include "calib/models/distortion.h"
#include "calib/pipeline/detail/planar_utils.h"  // for make_planar_view

namespace calib::pipeline {

using detail::make_planar_view;

namespace {

[[nodiscard]] auto build_point_lookup(const PlanarDetections& detections)
    -> std::unordered_map<std::string, const PlanarImageDetections*> {
    std::unordered_map<std::string, const PlanarImageDetections*> lookup;
    for (const auto& image : detections.images) {
        lookup.emplace(image.file, &image);
    }
    return lookup;
}

[[nodiscard]] auto to_dual_camera(const PinholeCamera<BrownConradyd>& cam)
    -> PinholeCamera<DualDistortion> {
    DualDistortion dual;
    dual.forward = cam.distortion.coeffs;
    dual.inverse = invert_brown_conrady(cam.distortion.coeffs);
    return PinholeCamera<DualDistortion>(cam.kmtx, dual);
}

}  // namespace

[[nodiscard]] auto compute_views(const StereoPairConfig& cfg,
                                 const PlanarDetections& reference_detections,
                                 const PlanarDetections& target_detections,
                                 const IntrinsicCalibrationOutputs& reference_intrinsics,
                                 const IntrinsicCalibrationOutputs& target_intrinsics,
                                 StereoCalibrationRunResult& result)
    -> std::vector<MulticamPlanarView> {
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

        auto reference_view = make_planar_view(*ref_it->second);
        auto target_view = make_planar_view(*tgt_it->second);
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
                                        const PlanarDetections& reference_detections,
                                        const PlanarDetections& target_detections,
                                        const IntrinsicCalibrationOutputs& reference_intrinsics,
                                        const IntrinsicCalibrationOutputs& target_intrinsics) const
    -> StereoCalibrationRunResult {
    StereoCalibrationRunResult result;
    result.requested_views = cfg.views.size();

    if (reference_intrinsics.refine_result.camera.distortion.coeffs.size() == 0 ||
        target_intrinsics.refine_result.camera.distortion.coeffs.size() == 0) {
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
        reference_intrinsics.refine_result.camera,
        target_intrinsics.refine_result.camera};

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
static auto compute_views(const MultiCameraRigConfig& cfg,
                          const std::unordered_map<std::string, PlanarDetections>& dets,
                          const std::unordered_map<std::string, IntrinsicCalibrationOutputs>& intr)
    -> std::vector<MulticamPlanarView> {
    // Build lookup tables for each sensor
    std::unordered_map<std::string,
                       std::unordered_map<std::string, const PlanarImageDetections*>>
        lookup;
    for (const auto& [sid, d] : dets) {
        std::unordered_map<std::string, const PlanarImageDetections*> map;
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
            auto view = make_planar_view(*img_det_it->second);
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
    const std::unordered_map<std::string, PlanarDetections>& detections_by_sensor,
    const std::unordered_map<std::string, IntrinsicCalibrationOutputs>& intrinsics_by_sensor) const
    -> MultiCameraCalibrationRunResult {
    MultiCameraCalibrationRunResult result;
    result.requested_views = cfg.views.size();
    result.sensors = cfg.sensors;

    // Validate intrinsics availability
    for (const auto& sid : cfg.sensors) {
        auto it = intrinsics_by_sensor.find(sid);
        if (it == intrinsics_by_sensor.end() ||
            it->second.refine_result.camera.distortion.coeffs.size() == 0) {
            throw std::runtime_error(
                "MultiCameraCalibrationFacade: intrinsics not available for sensor: " + sid);
        }
    }

    const auto views = compute_views(cfg, detections_by_sensor, intrinsics_by_sensor);
    result.used_views = views.size();
    if (views.empty()) {
        result.success = false;
        result.optimization.success = false;
        return result;
    }

    std::vector<PinholeCamera<BrownConradyd>> init_cameras;
    init_cameras.reserve(cfg.sensors.size());
    for (const auto& sid : cfg.sensors) {
        init_cameras.push_back(intrinsics_by_sensor.at(sid).refine_result.camera);
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
