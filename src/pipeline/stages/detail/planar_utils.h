#pragma once

#include <Eigen/Geometry>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "calib/pipeline/handeye.h"
#include "calib/pipeline/planar_intrinsics.h"

namespace calib::pipeline::detail {

struct SensorDetectionsIndex final {
    const planar::PlanarDetections* detections{nullptr};
    std::unordered_map<std::string, const planar::PlanarImageDetections*> image_lookup;
};

const planar::CameraConfig* find_camera_config(const planar::PlanarCalibrationConfig& cfg,
                                               const std::string& camera_id);

std::unordered_map<std::string, SensorDetectionsIndex> build_sensor_index(
    const std::vector<planar::PlanarDetections>& detections);

PlanarView make_planar_view(const planar::PlanarImageDetections& detections,
                            const planar::CalibrationOutputs& outputs);

Eigen::Isometry3d average_isometries(const std::vector<Eigen::Isometry3d>& poses);

const HandEyeRigConfig* find_handeye_rig(const HandEyePipelineConfig& cfg,
                                         const std::string& rig_id);

}  // namespace calib::pipeline::detail
