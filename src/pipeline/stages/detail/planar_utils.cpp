#include "planar_utils.h"

#include <algorithm>

#include "calib/estimation/common/se3_utils.h"

namespace calib::pipeline::detail {

const planar::CameraConfig* find_camera_config(const planar::PlanarCalibrationConfig& cfg,
                                               const std::string& camera_id) {
    const auto it = std::find_if(cfg.cameras.begin(), cfg.cameras.end(),
                                 [&](const planar::CameraConfig& cam) {
                                     return cam.camera_id == camera_id;
                                 });
    return it == cfg.cameras.end() ? nullptr : &(*it);
}

namespace {

std::unordered_map<std::string, const planar::PlanarImageDetections*>
build_point_lookup(const planar::PlanarDetections& detections) {
    std::unordered_map<std::string, const planar::PlanarImageDetections*> lookup;
    for (const auto& image : detections.images) {
        lookup.emplace(image.file, &image);
    }
    return lookup;
}

}  // namespace

std::unordered_map<std::string, SensorDetectionsIndex>
build_sensor_index(const std::vector<planar::PlanarDetections>& detections) {
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

PlanarView make_planar_view(const planar::PlanarImageDetections& detections,
                            const planar::CalibrationOutputs& outputs) {
    PlanarView view;
    view.reserve(detections.points.size());
    for (const auto& point : detections.points) {
        PlanarObservation obs;
        obs.object_xy = Eigen::Vector2d((point.local_x - outputs.point_center[0]) * outputs.point_scale,
                                        (point.local_y - outputs.point_center[1]) * outputs.point_scale);
        obs.image_uv = Eigen::Vector2d(point.x, point.y);
        view.push_back(std::move(obs));
    }
    return view;
}

Eigen::Isometry3d average_isometries(const std::vector<Eigen::Isometry3d>& poses) {
    if (poses.empty()) {
        return Eigen::Isometry3d::Identity();
    }
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_sum(0.0, 0.0, 0.0, 0.0);
    for (const auto& pose : poses) {
        translation += pose.translation();
        Eigen::Quaterniond q(pose.linear());
        if (q_sum.coeffs().dot(q.coeffs()) < 0.0) {
            q.coeffs() *= -1.0;
        }
        q_sum.coeffs() += q.coeffs();
    }
    translation /= static_cast<double>(poses.size());
    q_sum.normalize();
    Eigen::Isometry3d avg = Eigen::Isometry3d::Identity();
    avg.linear() = q_sum.toRotationMatrix();
    avg.translation() = translation;
    return avg;
}

const HandEyeRigConfig* find_handeye_rig(const HandEyePipelineConfig& cfg,
                                         const std::string& rig_id) {
    const auto it = std::find_if(cfg.rigs.begin(), cfg.rigs.end(),
                                 [&](const HandEyeRigConfig& rig) { return rig.rig_id == rig_id; });
    return it == cfg.rigs.end() ? nullptr : &(*it);
}

}  // namespace calib::pipeline::detail
