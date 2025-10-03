#pragma once

#include <Eigen/Geometry>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "calib/pipeline/facades/handeye.h"
#include "calib/pipeline/facades/intrinsics.h"

namespace calib::pipeline::detail {

struct SensorDetectionsIndex final {
    const PlanarDetections* detections{nullptr};
    std::unordered_map<std::string, const PlanarImageDetections*> image_lookup;
};

auto find_camera_config(const IntrinsicCalibrationConfig& cfg, const std::string& camera_id)
    -> const CameraConfig*;

[[nodiscard]] auto build_sensor_index(const std::vector<PlanarDetections>& detections)
    -> std::unordered_map<std::string, SensorDetectionsIndex>;

[[nodiscard]] auto make_planar_view(const PlanarImageDetections& detections) -> PlanarView;

[[nodiscard]] auto average_isometries(const std::vector<Eigen::Isometry3d>& poses)
    -> Eigen::Isometry3d;

const HandEyeRigConfig* find_handeye_rig(const HandEyePipelineConfig& cfg,
                                         const std::string& rig_id);

}  // namespace calib::pipeline::detail
