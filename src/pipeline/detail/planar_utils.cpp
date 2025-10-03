#include "calib/pipeline/detail/planar_utils.h"

#include <algorithm>

#include "calib/estimation/common/se3_utils.h"

namespace calib::pipeline::detail {

const CameraConfig* find_camera_config(const IntrinsicCalibrationConfig& cfg,
                                       const std::string& camera_id) {
    const auto it =
        std::find_if(cfg.cameras.begin(), cfg.cameras.end(),
                     [&](const CameraConfig& cam) { return cam.camera_id == camera_id; });
    return it == cfg.cameras.end() ? nullptr : &(*it);
}

namespace {

std::unordered_map<std::string, const PlanarImageDetections*> build_point_lookup(
    const PlanarDetections& detections) {
    std::unordered_map<std::string, const PlanarImageDetections*> lookup;
    for (const auto& image : detections.images) {
        lookup.emplace(image.file, &image);
    }
    return lookup;
}

}  // namespace

std::unordered_map<std::string, SensorDetectionsIndex> build_sensor_index(
    const std::vector<PlanarDetections>& detections) {
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

auto make_planar_view(const PlanarImageDetections& detections) -> PlanarView {
    PlanarView view(detections.points.size());
    std::transform(detections.points.begin(), detections.points.end(), view.begin(),
                   [&](const auto& point) -> PlanarObservation {
                       return {{point.local_x, point.local_y}, {point.x, point.y}};
                   });
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
